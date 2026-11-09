import os
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from monai.data import Dataset, DataLoader
from data.utils import IndexedDataset, load_data_splits_json
from data.transform import make_eval_transform
from models.utils import str2bool
from model.encoder_decoder import DINOv3Encoder
from model.seg_cls import get_backbone_out_indices,MultimodalMRIClassHead, MultimodalMRISegHead
from evaluation.eval import evaluate
from peft import LoraConfig, get_peft_model
from train import resolve_task_selection

## select the labels used for supervised_learning
MOLECULAR_TASKS = ("IDH", "1p19q", "MGMT", "TERT", "ATRX")
MOLECULAR_TASK_TO_INDEX = {task: idx for idx, task in enumerate(MOLECULAR_TASKS)}

SUBTYPE_CLASSES = {
    0: "astrocytoma_IDH_mutant",
    1: "glioblastoma_IDH_wildtype",
    2: "oligodendroglioma_IDH_mutant_1p19q_codeleted",
}
NUM_SUBTYPES = len(SUBTYPE_CLASSES)

class CombinedModel(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        classifier_head: Optional[nn.Module],
        segment_head: Optional[nn.Module],
        use_cls: bool = True,
        use_seg: bool = True,
    ):
        super().__init__()
        if use_cls and classifier_head is None:
            raise ValueError("classifier_head is required when use_cls=True.")
        if use_seg and segment_head is None:
            raise ValueError("segment_head is required when use_seg=True.")
        self.feature_model = feature_model
        self.classifier_head = classifier_head
        self.segment_head = segment_head
        self.use_cls = use_cls
        self.use_seg = use_seg

    def forward(self, data):
        imgs = data["image"]

        features = self.feature_model(data)
        cls_out = self.classifier_head(features) if self.use_cls else None
        seg_out = self.segment_head(imgs, features) if self.use_seg else None
        return cls_out, seg_out


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dinov3_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    log_file = output_dir / "test.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def choose_device(cuda_id: str) -> torch.device:
    # Use one GPU when available; otherwise fall back to CPU.
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
        return torch.device("cuda")
    return torch.device("cpu")


def torch_load_cpu(path: Path) -> Any:
    # weights_only is unavailable in older PyTorch versions.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj and all(
        isinstance(k, str) for k in obj.keys()
    )

def looks_like_tensor_state_dict(obj: Any) -> bool:
    if not is_state_dict(obj):
        return False
    tensor_like = 0
    for value in obj.values():
        if torch.is_tensor(value):
            tensor_like += 1
    return tensor_like > 0


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Extract model weights from common PyTorch checkpoint layouts.

    Supports:
    - torch.save(model.state_dict())
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - {"model_state_dict": state_dict}
    - {"module": state_dict}
    - {"feature_model": ..., "classifier_head": ..., "segment_head": ...}
    """
    if looks_like_tensor_state_dict(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a valid dict/state_dict object.")

    split_keys = ["feature_model", "classifier_head", "segment_head"]
    present_split_keys = [key for key in split_keys if key in checkpoint]
    if "feature_model" in present_split_keys:
        merged = {}
        for module_name in present_split_keys:
            module_state = checkpoint[module_name]
            if not looks_like_tensor_state_dict(module_state):
                raise ValueError(f"Checkpoint key '{module_name}' is not a state_dict.")
            for key, value in module_state.items():
                merged[f"{module_name}.{key}"] = value
        return merged

    candidate_keys = [
        "model",
        "state_dict",
        "model_state_dict",
        "module",
        "net",
        "network",
        "ema_state_dict",
    ]

    for key in candidate_keys:
        if key in checkpoint:
            value = checkpoint[key]
            if looks_like_tensor_state_dict(value):
                return value
            if isinstance(value, dict):
                try:
                    return extract_state_dict(value)
                except ValueError:
                    pass

    raise ValueError(
        "Could not find model weights in checkpoint. "
        f"Available top-level keys: {list(checkpoint.keys())}"
    )


def load_deepspeed_zero_state_dict(
    ckpt_dir: Path,
    tag: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    return get_fp32_state_dict_from_zero_checkpoint(
        str(ckpt_dir),
        tag=tag,
    )

def strip_prefix_if_better(
    state_dict: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """
    Strip a prefix only when it increases overlap with the target model keys.
    This avoids accidentally damaging valid parameter names.
    """
    model_keys = set(model_state.keys())
    old_overlap = len(set(state_dict.keys()) & model_keys)

    stripped = {}
    changed = False
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value
            changed = True
        else:
            stripped[key] = value

    if not changed:
        return state_dict

    new_overlap = len(set(stripped.keys()) & model_keys)
    return stripped if new_overlap >= old_overlap else state_dict


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    model_state = model.state_dict()

    # Common wrappers from DDP, custom model containers, or saved submodules.
    for prefix in [
        "module.",
        "model.",
        "backbone.",
        "model_engine.",
        "module.model.",
    ]:
        state_dict = strip_prefix_if_better(state_dict, model_state, prefix)

    return state_dict

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    weight_tag: str,
    device: torch.device,
    logger: logging.Logger,
    strict: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint from a checkpoint root directory.

    Expected usage:
    - DeepSpeed:
        checkpoint_path = /path/to/best_checkpoint
        weight_tag = best_joint

    - Plain PyTorch:
        checkpoint_path = /path/to/best_checkpoint
        weight_tag = best_joint_model.pth
    """
    checkpoint_path = Path(checkpoint_path)
    load_path = checkpoint_path / weight_tag

    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"checkpoint_path must be a directory: {checkpoint_path}")

    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint target not found: {load_path}")

    if load_path.is_dir():
        logger.info(
            f"Loading DeepSpeed ZeRO checkpoint: root={checkpoint_path}, tag={weight_tag}"
        )
        state_dict = load_deepspeed_zero_state_dict(
            checkpoint_path,
            tag=weight_tag,
        )
        checkpoint_meta = {
            "checkpoint_dir": str(checkpoint_path),
            "weight_tag": weight_tag,
            "type": "deepspeed",
        }

    else:
        logger.info(f"Loading PyTorch checkpoint file: {load_path}")
        checkpoint = torch_load_cpu(load_path)
        state_dict = extract_state_dict(checkpoint)
        checkpoint_meta = checkpoint if isinstance(checkpoint, dict) else {}

    state_dict = normalize_state_dict_keys(state_dict, model)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    model.to(device=device, dtype=torch.bfloat16)

    logger.info(f"Checkpoint loaded with strict={strict}")
    logger.info(f"Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.missing_keys:
        logger.warning(f"First missing keys: {incompatible.missing_keys[:30]}")

    logger.info(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
    if incompatible.unexpected_keys:
        logger.warning(f"First unexpected keys: {incompatible.unexpected_keys[:30]}")

    return checkpoint_meta


def make_loader(files, batch_size: int, num_workers: int) -> DataLoader:
    dataset = Dataset(data=files, transform=make_eval_transform())
    dataset = IndexedDataset(dataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

def build_model(
    args,
    num_classes,
    device: torch.device,
    test_loader: DataLoader,
    logger: logging.Logger,
    use_cls: bool,
    use_seg: bool,
) -> nn.Module:
    logger.info("Loading DINOv3 backbone...")
    logger.info(f"Backbone name: {args.dinov3_backbone_name}")

    backbone_repo = os.path.join(os.getcwd(), "dinov3")
    if args.use_pretrained:
        logger.info(f"Using pretrained weights: {args.pretrained_weights}")
        load_weights = args.pretrained_weights
    else:
        logger.info("Skipping pretrained weights (training from scratch for the head/backbone).")
        load_weights = None

    kwargs = dict(
        repo_or_dir=backbone_repo,
        model=args.dinov3_backbone_name,
        source="local",
        pretrained=args.use_pretrained,
    )

    if load_weights is not None:
        kwargs["weights"] = load_weights

    DINOv3_backbone = torch.hub.load(**kwargs)

    if args.use_peft:
        if get_peft_model is None:
            raise RuntimeError("PEFT is not installed. Install peft or run without --use_peft.")
        lora_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            bias="none",
        )
        DINOv3_backbone = get_peft_model(DINOv3_backbone, lora_config)

    block_idxs = get_backbone_out_indices(DINOv3_backbone)
    logger.info(f"Backbone={args.dinov3_backbone_name}, "f"n_blocks={DINOv3_backbone.n_blocks}, "f"output_indices={block_idxs}")

    feature_model = DINOv3Encoder(
        DINOv3_backbone,
        block_idxs_list=block_idxs,
        mod_keys=args.Cls_input_mods
        # fine_tune=args.fine_tune or args.use_peft,
    ).to(device)

    # Build heads from one transformed sample because the head constructors need feature shapes.
    feature_model.eval()
    with torch.no_grad():
        sample = next(iter(test_loader))
        sample = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        sample_features = feature_model(sample)

    mod0 = sample_features[feature_model.mod_keys[0]]
    logger.info(f"Sample patch feature shape: {mod0[0][0].shape}")
    logger.info(f"Sample cls feature shape: {mod0[0][1].shape}")

    classifier_head = None
    if use_cls:
        classifier_head = MultimodalMRIClassHead(
            sample_F_all=sample_features,
            hidden_dim=feature_model.embed_dim,
            num_classes=num_classes,
            use_n_blocks=1,
            use_patch_token=args.use_patch_token,
            modal_fusion=args.mods_fusion,
            slice_pooling=args.slice_pool,
        ).to(device)

    segmentation_head = None
    if use_seg:
        segmentation_head = MultimodalMRISegHead(
            sample_F_all=sample_features,
            hidden_dim=feature_model.embed_dim,
        ).to(device)

    model = CombinedModel(
        feature_model=feature_model,
        classifier_head=classifier_head,
        segment_head=segmentation_head,
        use_cls=use_cls,
        use_seg=use_seg,
    ).to(device)

    return model

def save_test_outputs(
    output_dir: Path,
    test_metrics: Dict[str, Any],
    test_outputs: Dict[str, Any],
    threshold,
    logger: logging.Logger,
    task_type: str = "molecular",
    use_cls: bool = True,
    use_seg: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "test_metrics.json"
    confmat_path = output_dir / "test_confmat.json"

    confmat_dict = {}
    if use_cls:
        labels = np.asarray(test_outputs["cls_labels"]).astype(int)
        probs = np.asarray(test_outputs["cls_probs"]).astype(float)
        masks = np.asarray(test_outputs["cls_masks"]).astype(int)
        tasks = test_outputs["tasks"]
        if task_type == "molecular":
            if labels.ndim == 1:
                labels = labels[:, None]
            if probs.ndim == 1:
                probs = probs[:, None]
            if masks.ndim == 1:
                masks = masks[:, None]
            if labels.shape != probs.shape or masks.shape != labels.shape:
                raise ValueError(f"Classification output shape mismatch: labels={labels.shape}, probs={probs.shape}, masks={masks.shape}")
            if probs.shape[1] != len(tasks):
                raise ValueError(f"Task number mismatch: probs.shape[1]={probs.shape[1]}, len(tasks)={len(tasks)}")
            for task_idx, task_name in enumerate(tasks):
                valid = masks[:, task_idx].astype(bool)
                y_true, y_prob = labels[valid, task_idx], probs[valid, task_idx]
                task_threshold = float(threshold.get(task_name, 0.5)) if isinstance(threshold, dict) else float(threshold)
                confmat_dict[task_name] = {
                    "labels": [0, 1], "threshold": task_threshold, "n": int(valid.sum()), "missing": int((~valid).sum()),
                    "matrix": confusion_matrix(y_true, y_prob >= task_threshold, labels=[0, 1]).tolist(),
                }
        else:
            labels, masks = labels.reshape(-1), masks.reshape(-1).astype(bool)
            confmat_dict["subtype"] = {
                "labels": list(range(probs.shape[1])), "n": int(masks.sum()), "missing": int((~masks).sum()),
                "matrix": confusion_matrix(labels[masks], np.argmax(probs[masks], axis=1), labels=list(range(probs.shape[1]))).tolist(),
            }
        with confmat_path.open("w", encoding="utf-8") as f:
            json.dump(confmat_dict, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **({"best_val_threshold": threshold, "test_cls": test_metrics["cls"]} if use_cls else {}),
                **({"test_seg": test_metrics["seg"]} if use_seg else {}),
                "test_all": test_metrics,
            },
            f,
            indent=2,
        )

    if use_cls:
        logger.info(f"Confusion matrices saved to: {confmat_path}")
    logger.info(f"Test metrics saved to: {metrics_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU test script for DINOv3 MRI classification/segmentation")

    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_mode", type=str, choices=["seg_cls", "cls", "seg"], default="seg_cls")

    parser.add_argument("--mods_fusion", type=str, choices=["concat", "mean", "attention"], default="attention")
    parser.add_argument("--slice_pool", type=str, choices=["mean", "max", "attention"], default="attention")
    parser.add_argument(
        "--task_select",
        nargs="+",
        default=list(MOLECULAR_TASKS),
        help="Molecular task(s), e.g. IDH MGMT, or 'subtype' for WHO multiclass classification.",
    )

    parser.add_argument(
        "--Cls_input_mods",
        nargs="+",
        choices=["flair_3ch", "t1_3ch", "t1ce_3ch", "t2_3ch"],
        default=["flair_3ch", "t1_3ch", "t1ce_3ch", "t2_3ch"],
        help="Modalities to encode, in the supplied order."
    )
    parser.add_argument("--use_patch_token", type=str2bool, default=False)
    parser.add_argument("--use_peft", action="store_true", help="Use LoRA for DINOv3 backbone adaptation")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--dinov3_backbone_name", type=str, default="dinov3_vitb16")

    parser.add_argument("--thresholds", type=float, default=None)

    parser.add_argument(
        "--test_seg_vis_samples",
        type=int,
        default=8,
        help="Maximum number of segmentation PNGs; masks are saved for every case.",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="path/to/pretrained/weights.pth",
    )
    parser.add_argument(
        "--data_splits_dir",
        type=str,
        help="Path to data_splits.json generated by the train.py.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="path/to/checkpoint",
        help="Path to .pth/.pt checkpoint file or DeepSpeed ZeRO checkpoint directory.",
    )
    parser.add_argument(
    "--weight_tag",
    type=str,
    default="best_joint",
    help="DeepSpeed checkpoint tag, e.g. best_cls, best_seg, best_joint./Torch tag. e.g. best_cls_model.pth, best_seg_model.pth, best_joint_model.pth",
)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path/to/output",
        help="Directory for test logs and result JSON files.",
    )
    parser.add_argument(
        "--strict_load",
        type=str2bool,
        default=False,
        help="Use strict=True for model.load_state_dict.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    use_cls = args.train_mode in {"seg_cls", "cls"}
    use_seg = args.train_mode in {"seg_cls", "seg"}

    output_dir = Path(args.output_dir) / "test_results"
    logger = setup_logger(output_dir)

    device = choose_device(args.cuda)
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Training mode: {args.train_mode}, use_cls={use_cls}, use_seg={use_seg}")

    checkpoint_path = Path(args.checkpoint)
    logger.info(f"Checkpoint selection: {args.weight_tag}")
    
    logger.info(f"Loading validation/test splits from: {args.data_splits_dir}")
    data_splits_file = Path(args.data_splits_dir) / "data_splits.json"
    splits = load_data_splits_json(data_splits_file)
    test_files = splits["test"]
    logger.info(f"Test samples: {len(test_files)}")
    test_loader = make_loader(test_files, args.batch_size, args.num_workers)

    ## 09.15 ------------------
    task_config = resolve_task_selection(args.task_select)
    if use_cls:
        logger.info(
            f"Task selection: type={task_config['task_type']}, tasks={task_config['tasks']}, "
            f"indices={task_config['task_indices']}, num_outputs={task_config['num_outputs']}"
        )
    task_type = task_config["task_type"]
    tasks = task_config["tasks"]
    task_indices = task_config["task_indices"]
    num_outputs = task_config["num_outputs"] 
    # ------------------------------

    logger.info("Building model...")
    model = build_model(args, num_outputs, device, test_loader, logger, use_cls, use_seg)

    logger.info("Loading checkpoint...")
    load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        logger=logger,
        strict=args.strict_load,
        weight_tag=args.weight_tag,
    )
    model.eval()

    with torch.inference_mode():
        logger.info("Running test evaluation...")
        test_metrics, test_outputs, test_summary = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            num_tasks=num_outputs,
            tasks=tasks,
            return_outputs=True,
            model_engine=None,
            task_type=task_type,
            task_indices=task_indices,
            use_cls=use_cls,
            use_seg=use_seg,
        )

    logger.info(f"Test metrics: {test_metrics}")
    save_test_outputs(
        output_dir=output_dir,
        test_metrics=test_metrics,
        test_outputs=test_outputs,
        logger=logger,
        task_type=task_type,
        use_cls=use_cls,
        use_seg=use_seg,
    )

    logger.info("Test evaluation complete.")

if __name__ == "__main__":
    main()
