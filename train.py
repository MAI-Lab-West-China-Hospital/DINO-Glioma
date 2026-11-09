import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data.utils import IndexedDataset, save_data_splits_json, build_multidataset_splits, load_data_splits_json
from data.transform import make_train_transform, make_eval_transform
from evaluation.eval import evaluate
import deepspeed
from peft import LoraConfig, get_peft_model
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from monai.data import Dataset, DataLoader, PersistentDataset
from model.encoder_decoder import DINOv3Encoder
from model.seg_cls import get_backbone_out_indices, MultimodalMRIClassHead, MultimodalMRISegHead
from losses.losses import DC_and_BCE_loss, WeightedTverskyLoss, masked_bce_loss
from helpers.ckpt import update_best_checkpoints, format_metric
import pandas as pd

def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    logger = logging.getLogger("dinov3_cls")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def log_flush(logger, msg, level="info"):
    log_fn = getattr(logger, level, logger.info)
    log_fn(msg)
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def broadcast_object(obj: Any) -> Any:
    if not is_distributed():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def prepare_deepspeed_config(
    config_path: str,
    batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))

    desired_micro = batch_size
    desired_accum = max(1, grad_accum_steps)
    desired_train = desired_micro * desired_accum * max(1, world_size)

    for key, desired in (
        ("train_micro_batch_size_per_gpu", desired_micro),
        ("gradient_accumulation_steps", desired_accum),
        ("train_batch_size", desired_train),
    ):
        existing = config.get(key)
        if existing is not None and existing != desired:
            logger.info(f"[DeepSpeed] Overriding {key}={existing} -> {desired}")
        config[key] = desired

    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False

def log_deepspeed_config(logger: logging.Logger, ds_config, model_engine=None):
    """安全地记录 DeepSpeed 配置 + 实际生效信息（兼容不同版本）"""
    logger.info("------DeepSpeed Configuration------")

    def _get_engine_config_dict(engine):
        if engine is None:
            return None
        for name in ("config", "_config", "ds_config"):
            cfg = getattr(engine, name, None)
            if isinstance(cfg, dict):
                return cfg
        return None

    engine_cfg = _get_engine_config_dict(model_engine)
    cfg = engine_cfg or ds_config

    logger.info(f"  Micro batch per GPU: {cfg.get('train_micro_batch_size_per_gpu', 'N/A')}")
    logger.info(f"  Gradient accumulation: {cfg.get('gradient_accumulation_steps', 'N/A')}")

    fp16_enabled = cfg.get("fp16", {}).get("enabled", False)
    bf16_enabled = cfg.get("bf16", {}).get("enabled", False)
    precision = "FP16" if fp16_enabled else ("BF16" if bf16_enabled else "FP32")
    logger.info(f"  Precision (config): {precision}")

    param_dtype = None
    if model_engine is not None:
        try:

            param_dtype = next(model_engine.module.parameters()).dtype
        except Exception:
            try:
                param_dtype = next(model_engine.parameters()).dtype
            except Exception:
                pass
    logger.info(f"  Engine param dtype: {param_dtype}")

    # ZeRO 配置
    zero_config = cfg.get("zero_optimization", {})
    logger.info(f"  ZeRO stage: {zero_config.get('stage', 'N/A')}")

    if isinstance(zero_config, dict):
        if "offload_optimizer" in zero_config:
            logger.info(f"  Offload optimizer: {zero_config['offload_optimizer'].get('device', 'N/A')}")
        if "offload_param" in zero_config:
            logger.info(f"  Offload params: {zero_config['offload_param'].get('device', 'N/A')}")

    # Activation checkpointing
    act_ckpt = cfg.get("activation_checkpointing", {})
    if isinstance(act_ckpt, dict) and act_ckpt:
        logger.info(
            "  Activation checkpointing: "
            f"partition={act_ckpt.get('partition_activations', False)}, "
            f"cpu={act_ckpt.get('cpu_checkpointing', False)}"
        )

    # 全局 batch size（用 world_size 说法更准确）
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        global_batch = (
            cfg.get("train_micro_batch_size_per_gpu", 1)
            * cfg.get("gradient_accumulation_steps", 1)
            * world_size
        )
        logger.info(f"  Global batch size: {global_batch} (world_size={world_size})")

    logger.info("---" * 15)

def log_cuda_peak(tag: str, device: torch.device, logger: Optional[logging.Logger] = None, reset: bool = False) -> None:
    if not torch.cuda.is_available():
        return
    if reset:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            torch.cuda.reset_peak_memory_stats()
    alloc = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    msg = f"[debug_mem] {tag} max_memory_allocated={alloc} max_memory_reserved={reserved}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def get_engine_mp_dtype(model_engine: Optional[Any]) -> Optional[torch.dtype]:
    if model_engine is None:
        return None
    if getattr(model_engine, "fp16_enabled", lambda: False)():
        return torch.float16
    if getattr(model_engine, "bfloat16_enabled", lambda: False)():
        return torch.bfloat16
    try:
        return next(model_engine.parameters()).dtype
    except StopIteration:
        return None

def maybe_cast_inputs_for_engine(data: torch.Tensor, model_engine: Optional[Any]) -> torch.Tensor:
    mp_dtype = get_engine_mp_dtype(model_engine)
    if mp_dtype is None:
        return data
    if data.dtype != mp_dtype:
        return data.to(dtype=mp_dtype)
    return data

def maybe_configure_deepspeed_activation_checkpointing(
    backbone: nn.Module,
    ds_config: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    if ds_config is None:
        return
    act_cfg = ds_config.get("activation_checkpointing", {})
    if not isinstance(act_cfg, dict) or not act_cfg:
        return
    if deepspeed is None:
        logger.warning("[DeepSpeed] activation_checkpointing set but DeepSpeed is unavailable.")
        return
    try:
        from deepspeed import checkpointing as ds_checkpointing
    except Exception as exc:
        logger.warning(f"[DeepSpeed] activation_checkpointing import failed: {exc}")
        return

    try:
        ds_checkpointing.configure(
            partition_activations=act_cfg.get("partition_activations", False),
            contiguous_memory_optimization=act_cfg.get("contiguous_memory_optimization", False),
            cpu_checkpointing=act_cfg.get("cpu_checkpointing", False),
            profile=act_cfg.get("profile", False),
        )
    except TypeError:
        try:
            ds_checkpointing.configure(deepspeed_config=ds_config)
        except TypeError:
            try:
                ds_checkpointing.configure(None, ds_config)
            except Exception as exc:
                logger.warning(f"[DeepSpeed] activation_checkpointing configure failed: {exc}")
                return
        except Exception as exc:
            logger.warning(f"[DeepSpeed] activation_checkpointing configure failed: {exc}")
            return

    if hasattr(backbone, "set_checkpointing_fn"):
        backbone.set_checkpointing_fn(ds_checkpointing.checkpoint)
        logger.info("[DeepSpeed] Using DeepSpeed activation checkpointing function.")
    else:
        setattr(backbone, "_checkpoint_fn", ds_checkpointing.checkpoint)        

def save_jsonl(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# --- monkey patch for local URL loading + disable weights_only ---
def _patched_load_state_dict_from_url(url, *args, **kwargs):
    if url.startswith("file://"):
        # Strip file:// so torch.load treats it as a local path.
        url = url.replace("file://", "")
    return torch.load(url, map_location="cpu", weights_only=False)

torch.hub.load_state_dict_from_url = _patched_load_state_dict_from_url

## select the labels used for supervised_learning
MOLECULAR_TASKS = ("IDH", "1p19q", "MGMT", "TERT", "ATRX")
MOLECULAR_TASK_TO_INDEX = {task: idx for idx, task in enumerate(MOLECULAR_TASKS)}

SUBTYPE_CLASSES = {
    0: "astrocytoma_IDH_mutant",
    1: "glioblastoma_IDH_wildtype",
    2: "oligodendroglioma_IDH_mutant_1p19q_codeleted",
}
NUM_SUBTYPES = len(SUBTYPE_CLASSES)


def resolve_task_selection(task_select):
    selected = [task_select] if isinstance(task_select, str) else list(task_select)
    if not selected:
        raise ValueError("task_select must contain at least one task.")

    if selected == ["subtype"]:
        return {
            "task_type": "subtype",
            "tasks": ["subtype"],
            "task_indices": None,
            "num_outputs": NUM_SUBTYPES,
        }

    if "subtype" in selected:
        raise ValueError("'subtype' cannot be combined with molecular tasks.")

    invalid = [task for task in selected if task not in MOLECULAR_TASK_TO_INDEX]
    if invalid:
        raise ValueError(f"Unknown molecular task(s): {invalid}")

    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate tasks are not allowed: {selected}")

    return {
        "task_type": "molecular",
        "tasks": selected,
        "task_indices": [MOLECULAR_TASK_TO_INDEX[task] for task in selected],
        "num_outputs": len(selected),
    }
 
class CombinedModel(nn.Module):
    def __init__(
        self,
        feature_model: nn.Module,
        classifier_head: Optional[nn.Module],
        segment_head: Optional[nn.Module],
        freeze_feature: bool = True,
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
        self.freeze_feature = freeze_feature
        self.use_cls = use_cls
        self.use_seg = use_seg

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbone should always stay in eval mode
        if self.freeze_feature:
            self.feature_model.eval()
        return self

    def forward(self, data):
    
        imgs = data["image"]   

        if self.freeze_feature:
            with torch.no_grad():
                features = self.feature_model(data)
        else:
            features = self.feature_model(data)

        # cls_out = self.classifier_head(features, seg=data.get("seg")) if self.use_cls else None
        cls_out = self.classifier_head(features) if self.use_cls else None
        seg_out = self.segment_head(imgs, features) if self.use_seg else None
        return cls_out, seg_out

def train_one_epoch(
    model: CombinedModel,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    data_loader: DataLoader,
    device: torch.device,
    task_type: str = "molecular",
    task_indices: Optional[List[int]] = None,
    use_cls: bool = True,
    use_seg: bool = True,
    cls_weight: float = 1.0,
    seg_weight: float = 1.0,
    seg_class_weights: Tuple[float, float, float] = (1.0, 1.0, 2.0),
    seg_tversky_alpha: Tuple[float, float, float] = (0.5, 0.5, 0.7),
    seg_manual_weight: float = 1.0,
    seg_pseudo_weight: float = 1.0,
    model_engine: Optional[Any] = None,    
    logger: Optional[logging.Logger] = None
) -> float:
    
    # access the underlying FullModel regardless of DeepSpeed wrapping
    inner: CombinedModel = model_engine.module if model_engine is not None else model
    inner.train()
    if inner.freeze_feature:
        # correctly freeze backbone in both DeepSpeed and native mode
        inner.feature_model.eval()
    model_dtype = next(inner.parameters()).dtype
 
    total_loss = total_cls = total_seg = 0.0

    subtype_cls_fn = nn.CrossEntropyLoss()

    seg_fn = DC_and_BCE_loss(
        bce_kwargs={},
        soft_dice_kwargs={
            "batch_dice": False,
            "do_bg": True,
            "smooth": 1e-5,
            "ddp": False,
            "alpha": seg_tversky_alpha,
            "beta": tuple(1.0 - value for value in seg_tversky_alpha),
            "class_weights": seg_class_weights,
            "reduction": "none"
        },
        weight_ce=1,
        weight_dice=1,
        use_ignore_label=False,
        dice_class=WeightedTverskyLoss,
        class_weights=seg_class_weights,
        reduction="none"
    )

    for batch_data in tqdm(data_loader, desc="Train", leave=False):
        _MULTI_TASK_KEYS = {"label", "label_mask"} if use_cls else set()
        _FLOAT32_KEYS = ({"label", "label_mask"} if use_cls else set()) | ({"seg"} if use_seg else set())
        for k, v in batch_data.items():
            if (not use_cls and k in {"label", "label_mask", "subtype", "subtype_mask"}) or (not use_seg and k in {"seg", "seg_label_type"}):
                continue
            if isinstance(v, list) and k in _MULTI_TASK_KEYS:
                # label / label_mask: list of tensors [B] -> [B, T]
                batch_data[k] = torch.stack(v, dim=1).to(device=device, dtype=torch.float32)
            elif isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    dtype = torch.float32 if k in _FLOAT32_KEYS else model_dtype
                    batch_data[k] = v.to(device=device, dtype=dtype, non_blocking=True)
                else:
                    batch_data[k] = v.to(device=device, non_blocking=True)

        if use_seg:
            targets = batch_data["seg"]
            manual_mask = torch.tensor(
                [str(x).strip().lower() == "manual" for x in batch_data["seg_label_type"]],
                device=device, dtype=torch.bool,
            )
            seg_weights = torch.where(
                manual_mask,
                torch.tensor(seg_manual_weight, device=device, dtype=targets.dtype),
                torch.tensor(seg_pseudo_weight, device=device, dtype=targets.dtype),
            )

        # forward
        if model_engine is not None:
            cls_out, seg_out = model_engine(batch_data)
        else:
            cls_out, seg_out = model(batch_data)

        if use_cls:
            B, T = cls_out.shape
            if task_type == "molecular":
                indices = list(range(T)) if task_indices is None else task_indices
                if len(indices) != T:
                    raise ValueError(f"Expected {T} molecular task indices, got {indices}.")
                labels = batch_data["label"][:, indices].to(device=device)
                label_mask = batch_data["label_mask"][:, indices].to(device=device)
                cls_loss = masked_bce_loss(cls_out, labels, label_mask)
            elif task_type == "subtype":
                subtype = batch_data["subtype"].view(-1).to(device=device, dtype=torch.long)
                valid = batch_data["subtype_mask"].view(-1).to(device=device) == 1
                if subtype.shape[0] != B:
                    raise ValueError(f"Subtype target batch size {subtype.shape[0]} does not match logits batch size {B}.")
                invalid = valid & ((subtype < 0) | (subtype >= T))
                if invalid.any():
                    raise ValueError(f"Valid subtype targets must be in [0, {T - 1}], got {subtype[invalid].tolist()}.")
                cls_loss = subtype_cls_fn(cls_out[valid], subtype[valid]) if valid.any() else cls_out.sum() * 0.0
            else:
                raise ValueError(f"Unknown task_type '{task_type}'.")

        if use_seg:
            seg_loss_per_sample = seg_fn(seg_out, targets)
            seg_loss = (seg_loss_per_sample * seg_weights).mean()

        if use_cls and use_seg:
            train_loss = cls_weight * cls_loss + seg_weight * seg_loss
        elif use_cls:
            train_loss = cls_loss
        else:
            train_loss = seg_loss

        # backward
        if model_engine is not None:
            model_engine.backward(train_loss)
            model_engine.step()
        else:
            if optimizer is None:
                raise ValueError("optimizer must be provided when not using DeepSpeed")
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        total_loss += train_loss.item()
        if use_cls:
            total_cls += cls_loss.item()
        if use_seg:
            total_seg += seg_loss.item()

    if scheduler is not None and model_engine is None:
        scheduler.step()

    n = max(1, len(data_loader))

    return {
        "loss":     total_loss / n,
        "cls_loss": total_cls  / n,
        "seg_loss": total_seg  / n,
    }

def main():
    parser = argparse.ArgumentParser(description="DINOv3 linear classification (train/val/test)")
    parser.add_argument('--cuda', default='0', type=str, help='gpu id')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed config JSON path")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps (DeepSpeed)")
    
    ## load split_data or gnerate split_data & save
    parser.add_argument(
        "--data_splits_dir",
        type=str,
        default="none",
        help="Path to the exisiting or a specific data_splits.json generated before."
    )
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

    parser.add_argument("--train_mode", type=str, choices=["seg_cls", "cls", "seg"], default="seg_cls")
    parser.add_argument("--batch_size", type=int, default=2) #2
    parser.add_argument("--num_workers", type=int, default=8) #8-original 0-debug260121
    parser.add_argument("--epochs", type=int, default=10, help="epochs for final training with best hp")
    parser.add_argument("--use_patch_token", type=str2bool,default=False, help="Whether use patch token for classifier")
    parser.add_argument("--mods_fusion", type=str, choices=["concat", "mean", "attention"], default="attention")
    parser.add_argument("--slice_pool", type=str, choices=["mean", "max", "attention"], default="attention")
    parser.add_argument("--use_pretrained", type=str2bool, default=True, help="Whether to load pretrained backbone weights")
    parser.add_argument("--dinov3_backbone_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_weights", type=str, default="/Data3/generate_2d_dataset/DINOv3_Medical/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument(
        "--seg_class_weights", type=float, nargs=3, default=[1.0, 1.0, 2.0],
        metavar=("TC", "WT", "ET"),
        help="Per-region loss weights in target order TC WT ET."
        )
    parser.add_argument(
        "--seg_tversky_alpha", type=float, nargs=3, default=[0.5, 0.5, 0.7],
        metavar=("TC", "WT", "ET"),
        help="False-positive Tversky weights in target order TC WT ET; beta is 1-alpha.",
        )
    parser.add_argument("--seg_manual_weight", type=float, default=1.0)
    parser.add_argument("--seg_pseudo_weight", type=float, default=0.6)

    parser.add_argument("--output_dir", type=str, default="path/to/outputs")

    parser.add_argument("--fine_tune", action="store_true", help="Whether to fine-tune backbone")
    parser.add_argument("--use_peft", action="store_true", help="Use LoRA for DINOv3 backbone adaptation")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for DINO-backbone.")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Learning rate for head.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=0, help="epochs without main-metric improvement before stop")
    parser.add_argument("--confmat_interval", type=int, default=5, help="epochs between confusion-matrix logs")
    parser.add_argument("--val_seg_vis_samples", type=int, default=6, help="maximum validation segmentation cases saved for best_seg")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable model-internal grad checkpointing")    

    args = parser.parse_args()

    use_cls = args.train_mode in {"seg_cls", "cls"}
    use_seg = args.train_mode in {"seg_cls", "seg"}

    use_deepspeed = args.deepspeed
    if use_deepspeed and deepspeed is None:
        raise RuntimeError("DeepSpeed is not installed. Install it or run without --deepspeed.")
    
    if use_deepspeed:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_object_group = dist.new_group(backend="gloo") if is_distributed() else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = broadcast_object(timestamp)
    backbone_suffix = args.dinov3_backbone_name.split("dinov3_")[-1] if "dinov3_" in args.dinov3_backbone_name else ""
    folder_name = f"{timestamp}_{backbone_suffix}" if backbone_suffix else timestamp
    output_dir = Path(args.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        logger = setup_logger(output_dir)
    else:
        logger = logging.getLogger("dinov3_cls")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False

    # search_metrics_file = output_dir / "search_metrics.jsonl"
    ckpt_dir = output_dir / "best_checkpoint"
    tb_dir = output_dir / "tb"
    writer = SummaryWriter(log_dir=tb_dir) if is_main_process() else None
    logger.info(f"Device: {device}")
    logger.info(f"Args: {args}")
    logger.info(f"Training mode: {args.train_mode}, use_cls={use_cls}, use_seg={use_seg}")
    world_size = dist.get_world_size() if is_distributed() else 1

    ## ########################## Data ##########################################################################################
    logger.info("------Data info------")
    if args.data_splits_dir == "none":
        train_files, val_files, test_files = build_multidataset_splits(
            total_root="/Data3",
            dataset_names=["HX_Glioma", "UCSF_PDGM", "UPENN-GBM"],
            tasks=list(MOLECULAR_TASKS),
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42,
        )

        save_data_splits_json(
            out_dir=output_dir,
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            logger=logger
        ) 
    else:
        logger.info(f"Load the data_split from: {args.data_splits_dir}")
        data_splits_file = Path(args.data_splits_dir) / "data_splits.json"
        splits = load_data_splits_json(data_splits_file)
        train_files = splits["train"]
        val_files = splits["val"]

    ### add: select task type
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

    ### dataset,dataloader
    cache_dir = Path(output_dir) / "monai_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PersistentDataset(data=train_files, transform=make_train_transform(), cache_dir=cache_dir / "train")
    train_ds = IndexedDataset(train_ds)

    ## val_cache_dir = cache_dir / f"val_rank{get_rank()}" if is_distributed() else cache_dir / "val"
    ## val_ds = PersistentDataset(data=val_files, transform=make_eval_transform(), cache_dir=val_cache_dir)    
    val_ds = Dataset(data=val_files, transform=make_eval_transform())
    val_ds = IndexedDataset(val_ds)

    if is_distributed():
        rank = dist.get_rank()
        print(f"[rank{rank}] len(train_dataset)={len(train_ds)} type={type(train_ds)}", flush=True) 
        log_flush(logger, f"[rank{rank}] len(train_dataset)={len(train_ds)} type={type(train_ds)}")

        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        print(f"[rank{rank}] sampler.num_samples={train_sampler.num_samples} total_size={train_sampler.total_size}", flush=True)
        log_flush(logger, f"[rank{rank}] sampler.num_samples={train_sampler.num_samples} total_size={train_sampler.total_size}")
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
        # test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers
        )

        print(f"[rank{rank}] batch_size={train_loader.batch_size} drop_last={train_loader.drop_last}", flush=True)
        log_flush(logger, f"[rank{rank}] batch_size={train_loader.batch_size} drop_last={train_loader.drop_last}")
        dist.barrier()

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers
        )

    else:
        train_loader  = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader    = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # logger.info(f"Train/Val/Test sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")
    logger.info(f"Train/Val sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    logger.info("---" * 15)

    ## ########################## Feature model: extract cls tokens and patch tokens #########################################
    logger.info("Loading DINOv3 backbone...")
    logger.info(f"Using model: {args.dinov3_backbone_name}")
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
        source="local"
    )
    if load_weights is not None:
        kwargs["weights"] = load_weights

    DINOv3_backbone = torch.hub.load(**kwargs)

    ### FTT/LORA/FROZEN
    if args.use_peft and args.fine_tune:
        raise ValueError("--use_peft and --fine_tune are mutually exclusive. Please choose only one.")  
    if args.fine_tune:
        backbone_adaptation = "full"
    elif args.use_peft:
        backbone_adaptation = 'peft'
    else:
        backbone_adaptation = "frozen"

    # Gradient checkpointing is only useful when backbone gradients are required
    use_grad_ckpt = args.grad_ckpt and backbone_adaptation != "frozen"
    DINOv3_backbone.grad_checkpointing = use_grad_ckpt

    if backbone_adaptation == "peft":
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
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    elif backbone_adaptation == "full":
        DINOv3_backbone.requires_grad_(True)

    elif backbone_adaptation == "frozen":
        DINOv3_backbone.requires_grad_(False)

    logger.info(f"Backbone adaptation: {backbone_adaptation}")
    logger.info(f"Gradient checkpointing: {use_grad_ckpt}")

    backbone_trainable = sum(p.numel() for p in DINOv3_backbone.parameters() if p.requires_grad)
    backbone_total = sum(p.numel() for p in DINOv3_backbone.parameters())
    logger.info(f"Backbone trainable parameters: {backbone_trainable:,}/{backbone_total:,}")
    block_idxs = get_backbone_out_indices(DINOv3_backbone)
    logger.info(f"Backbone={args.dinov3_backbone_name}, "f"n_blocks={DINOv3_backbone.n_blocks}, "f"output_indices={block_idxs}")

    feature_model =  DINOv3Encoder(
        DINOv3_backbone, 
        block_idxs_list=block_idxs,
        mod_keys=args.Cls_input_mods
        )    
    feature_model.to(device)

    ## get sample to auto set the input of seg and cls heads
    with torch.no_grad():
        sample = next(iter(train_loader))
        sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in sample.items()}
        sample_encoder_features = feature_model(sample)
        mod0 = sample_encoder_features[feature_model.mod_keys[0]]
        if is_main_process():
            logger.info(f"Sample patch feature shape of each mod: {mod0[0][0].shape}")
            logger.info(f"Sample cls feature shape of each mod: {mod0[0][1].shape}")

    ## ------------------ define cls head ----------------------------------------------------------------------------------------
    classifier_head = None
    if use_cls:
        logger.info("Loading linear_cls head...")
        logger.info(f"---Using modal fusion: {args.mods_fusion}")
        logger.info(f"---Using slice pool: {args.slice_pool}")
        classifier_head = MultimodalMRIClassHead(
            sample_F_all=sample_encoder_features,
            hidden_dim=feature_model.embed_dim,
            num_classes=num_outputs,
            use_n_blocks=1,
            use_patch_token=False,
            modal_fusion=args.mods_fusion,
            slice_pooling=args.slice_pool,
            # tumor_slice=args.tumor_only
        ).to(device)

    ## ------------------ define seg head  ----------------------------------------------------------------------------------------
    segmentation_head = None
    if use_seg:
        logger.info("Loading seg head based on UNETRDecoder3D...")
        segmentation_head = MultimodalMRISegHead(
            sample_F_all=sample_encoder_features,
            hidden_dim=feature_model.embed_dim,
        ).to(device)

    # ### check the output of seg head
    # logits = segmentation_head(sample["image"], sample_encoder_features)
    # print("logits shape:", logits.shape) #torch.Size([2, 3, 128, 224, 224])
    # print(sample["image"].shape) ## torch.Size([2, 4, 128, 224, 224])
    # print("------set seg_head done------")

    ## Release or avoid the GPU sizing sample immediately after constructing the heads
    del mod0
    del sample_encoder_features
    del sample

    ds_config = None
    if use_deepspeed:
        ds_config = prepare_deepspeed_config(
            args.deepspeed_config,
            args.batch_size,
            args.grad_accum_steps,
            world_size,
            logger,
        )
        maybe_configure_deepspeed_activation_checkpointing(DINOv3_backbone, ds_config, logger)

    ############################################################################################################################
    logger.info(f"====== Training model in {args.train_mode} mode ======")

    freeze_feature = backbone_adaptation == "frozen"
    model = CombinedModel(
        feature_model=feature_model,
        classifier_head=classifier_head,
        segment_head=segmentation_head,
        freeze_feature=freeze_feature,
        use_cls=use_cls,
        use_seg=use_seg,
    )

    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("feature_model.")
    ]
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("feature_model.")
    ]
    optim_param_groups = []
    if head_params:
        optim_param_groups.append({"params": head_params, "lr": args.head_lr})
    if backbone_params:
        optim_param_groups.append({"params": backbone_params, "lr": args.backbone_lr})

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_param_ids = {id(parameter) for group in optim_param_groups for parameter in group["params"]}
    if optimizer_param_ids != {id(parameter) for parameter in trainable_params}:
        raise RuntimeError("Optimizer parameter groups do not match model trainable parameters.")
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected.")
    logger.info(f"Optimizer trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"Head lr={args.head_lr}, backbone lr={args.backbone_lr}")

    model_engine = None
    if use_deepspeed:
        ## 0831 scheduler
        updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
        total_num_steps = updates_per_epoch * args.epochs
        warmup_num_steps = min(total_num_steps, max(2, round(total_num_steps * args.warmup_ratio)))
        scheduler_params = ds_config["scheduler"]["params"]
        scheduler_params["total_num_steps"] = int(total_num_steps)
        scheduler_params["warmup_num_steps"] = int(warmup_num_steps)

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=optim_param_groups,
            config=ds_config
        )
        log_deepspeed_config(logger, ds_config, model_engine)
    else:
        # optimizer = optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0) ## use, before the 0515 
        optimizer = torch.optim.AdamW(optim_param_groups, lr=args.head_lr, weight_decay=1e-5)
        # total_steps = args.epochs * max(1, len(train_loader))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)    

    # ── Training state ───────────────────────────────────────────────────────
    patience = getattr(args, "early_stop_patience", 5)
    confmat_interval = getattr(args, "confmat_interval", 1)

    # Independent best scores for three checkpoint criteria
    best_scores: Dict[str, float] = {
        "cls":   -math.inf,
        "seg":   -math.inf,
        "joint": -math.inf,
    }
    epochs_since_improve = 0   # based on joint score for early stopping
    history: Dict[str, List[float]] = {}
    split_identifier = str(
        (output_dir / "data_splits.json")
        if args.data_splits_dir == "none"
        else (Path(args.data_splits_dir) / "data_splits.json")
    )
    checkpoint_metadata = {
        "mod_keys": list(feature_model.mod_keys),
        "task_mode": args.train_mode,
        "tasks": list(tasks),
        "task_indices": None if task_indices is None else list(task_indices),
        "backbone_name": args.dinov3_backbone_name,
        "backbone_adaptation": backbone_adaptation,
        "use_peft": bool(args.use_peft),
        "fine_tune": bool(args.fine_tune),
        "split_identifier": split_identifier
    }
 
    # ════════════════════════════════════════════════════════════════════════
    # Training loop
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(1, args.epochs + 1):
        if is_main_process():
            logger.info(f"[Train] Epoch {epoch}/{args.epochs} started")

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_losses = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader = train_loader,
            device=device,
            task_type=task_type,
            task_indices=task_indices,
            use_cls=use_cls,
            use_seg=use_seg,
            model_engine=model_engine,
            cls_weight=args.cls_weight,
            seg_weight=args.seg_weight,
            seg_class_weights=tuple(args.seg_class_weights),
            seg_tversky_alpha=tuple(args.seg_tversky_alpha),
            seg_manual_weight =args.seg_manual_weight,
            seg_pseudo_weight = args.seg_pseudo_weight,
            logger=logger
        )

        # ─────────────────────────────────────────────────────────────────────────────
        # Validation
        # ─────────────────────────────────────────────────────────────────────────────
        val_metrics, val_outputs = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            num_tasks=num_outputs,
            tasks=tasks,
            cls_threshold=0.5,
            return_outputs=True,
            model_engine=model_engine,
            task_type=task_type,
            task_indices=task_indices,
            use_cls=use_cls,
            use_seg=use_seg,
            object_group=eval_object_group,
        )

        should_stop = False
        improved_flags: Dict[str, bool] = {"cls": False, "seg": False, "joint": False,}

        if is_main_process():
            cls_m = val_metrics.get("cls", {})
            seg_m = val_metrics.get("seg", {})

            cls_macro = cls_m.get("macro_avg", {}) if use_cls and task_type == "molecular" else {
                "auc": cls_m.get("auc_macro"), "bal_acc": cls_m.get("bal_acc"),
                "f1": cls_m.get("f1_macro"), "acc": cls_m.get("acc"),
            }
            cls_micro = cls_m.get("micro_avg", {}) if use_cls and task_type == "molecular" else {}

            # ── Update best checkpoints ─────────────────────────────────────────────
            if not use_deepspeed:
                improved_flags = update_best_checkpoints(
                    epoch=epoch,
                    val_metrics=val_metrics,
                    best_scores=best_scores,
                    feature_model=feature_model,
                    classifier_head=classifier_head,
                    segment_head=segmentation_head,
                    ckpt_dir=ckpt_dir,
                    model_engine=None,
                    logger=logger,
                    task_type=task_type,
                    use_cls=use_cls,
                    use_seg=use_seg,
                    checkpoint_metadata=checkpoint_metadata,
                )

            # ── Log compact validation summary ──────────────────────────────────────
            if logger:
                if use_cls and use_seg and task_type == "molecular":
                    logger.info(
                        f"[Val] Epoch {epoch} | MacroAUC={format_metric(cls_macro.get('auc'))} "
                        f"MacroBalAcc={format_metric(cls_macro.get('bal_acc'))} MacroF1={format_metric(cls_macro.get('f1'))} | "
                        f"MicroAUC={format_metric(cls_micro.get('auc'))} MicroF1={format_metric(cls_micro.get('f1'))} | "
                        f"DiceMean={format_metric(seg_m.get('dice_mean'))}"
                    )
                elif use_cls and use_seg:
                    logger.info(
                        f"[Val] Epoch {epoch} | SubtypeMacroAUC={format_metric(cls_m.get('auc_macro'))} "
                        f"SubtypeBalAcc={format_metric(cls_m.get('bal_acc'))} "
                        f"SubtypeMacroF1={format_metric(cls_m.get('f1_macro'))} | "
                        f"DiceMean={format_metric(seg_m.get('dice_mean'))}"
                    )
                elif use_cls and task_type == "molecular":
                    logger.info(
                        f"[Val] Epoch {epoch} | MacroAUC={format_metric(cls_macro.get('auc'))} "
                        f"MacroBalAcc={format_metric(cls_macro.get('bal_acc'))} MacroF1={format_metric(cls_macro.get('f1'))} | "
                        f"MicroAUC={format_metric(cls_micro.get('auc'))} MicroF1={format_metric(cls_micro.get('f1'))}"
                    )
                elif use_cls:
                    logger.info(
                        f"[Val] Epoch {epoch} | SubtypeMacroAUC={format_metric(cls_m.get('auc_macro'))} "
                        f"SubtypeBalAcc={format_metric(cls_m.get('bal_acc'))} SubtypeMacroF1={format_metric(cls_m.get('f1_macro'))}"
                    )
                else:
                    logger.info(f"[Val] Epoch {epoch} | DiceMean={format_metric(seg_m.get('dice_mean'))}")

        # ── DeepSpeed checkpoint saving: all ranks must call together ───────────────
        if use_deepspeed:
            improved_flags = update_best_checkpoints(
                epoch=epoch,
                val_metrics=broadcast_object(val_metrics if is_main_process() else {}),
                best_scores=best_scores,
                feature_model=feature_model,
                classifier_head=classifier_head,
                segment_head=segmentation_head,
                ckpt_dir=ckpt_dir,
                model_engine=model_engine,
                logger=logger,
                task_type=task_type,
                use_cls=use_cls,
                use_seg=use_seg,
                checkpoint_metadata=checkpoint_metadata,
            )

        if is_main_process():  
            stop_key = "joint" if use_cls and use_seg else "cls" if use_cls else "seg"
            if improved_flags[stop_key]:
                epochs_since_improve = 0
            elif patience > 0:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    if logger:
                        logger.info(f"[Train] Early stopping: {patience} epochs without {stop_key} improvement.")
                    should_stop = True

        should_stop = broadcast_object(should_stop)

        if should_stop:
            break 
 
    if writer is not None and is_main_process():
        writer.close()
 
if __name__ == "__main__":
    main()

