import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.data import DataLoader
from tqdm import tqdm
from .compute_metrics import compute_multitask_binary_metrics, compute_multiclass_metrics

def is_distributed() -> bool:
    """是否处于分布式训练环境（torch.distributed 已初始化且 world_size > 1）"""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    """当前进程是否为主进程（rank 0）；单卡时始终返回 True"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _classification_probabilities_fp32(
    cls_logits: torch.Tensor,
    task_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detach logits, convert once to FP32, then apply exactly one activation."""
    logits_fp32 = cls_logits.detach().to(dtype=torch.float32)
    if task_type == "molecular":
        return logits_fp32, torch.sigmoid(logits_fp32)
    if task_type == "subtype":
        return logits_fp32, torch.softmax(logits_fp32, dim=1)
    raise ValueError(f"Unknown task_type {task_type!r}.")


@torch.no_grad()
def evaluate(
    model:           torch.nn.Module,
    data_loader:     DataLoader,
    device:          torch.device,
    seg_threshold:   float = 0.5,
    num_tasks:       int = 1,
    tasks:           Optional[List[str]] = None,
    cls_threshold:   Optional[Union[float, Dict[str, float]]] = 0.5,
    return_outputs:  bool = False,
    model_engine:    Optional[Any] = None,
    task_type:       str = "molecular",
    task_indices:    Optional[List[int]] = None,
    use_cls:         bool = True,
    use_seg:         bool = True,
    object_group:    Optional[Any] = None,
) -> Union[
    Dict[str, Dict[str, float]],
    Tuple[Dict[str, Dict[str, float]], Dict[str, Any]],
]:

    inner: torch.nn.Module = model_engine.module if model_engine is not None else model
    model_dtype = next(inner.parameters()).dtype
    inner.eval()


    total_samples: int = len(data_loader.dataset)

    if use_cls:
        if task_type not in {"molecular", "subtype"}:
            raise ValueError(f"Unknown task_type '{task_type}'.")
        if tasks is None:
            tasks = [f"task_{i}" for i in range(num_tasks)] if task_type == "molecular" else ["subtype"]
        if task_type == "molecular" and len(tasks) != num_tasks:
            raise ValueError(f"len(tasks)={len(tasks)} does not match num_tasks={num_tasks}.")
        if task_type == "subtype" and tasks != ["subtype"]:
            raise ValueError(f"Subtype evaluation expects tasks=['subtype'], got {tasks}.")

    post_seg = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=seg_threshold),
    ]) if use_seg else None

    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    ) if use_seg else None

    seg_preds_list:  List = []
    seg_labels_list: List = []
    seg_images_list: List = []
    seg_indices_list: List[int] = []
    seg_stats_local: List[Dict[str, Any]] = []

    cls_probs:       List[List[float]] = []
    cls_logits_fp32: List[List[float]] = []
    cls_labels:      List[List[int]] = []
    cls_masks:       List[List[int]] = []
    cls_source_dtypes: List[str] = []

    all_indices:     List[int] = []

    start_time = time.time()

    for batch_data in tqdm(data_loader, desc="Eval", leave=False):

        _MULTI_TASK_KEYS = {"label", "label_mask"} if use_cls else set()
        _FLOAT32_KEYS = ({"label", "label_mask"} if use_cls else set()) | ({"seg"} if use_seg else set())
        for k, v in batch_data.items():
            if (not use_cls and k in {"label", "label_mask", "subtype", "subtype_mask"}) or (not use_seg and k == "seg"):
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

        indices = batch_data["index"]
        if use_seg:
            seg_gt = batch_data["seg"]
        if use_cls:
            if task_type == "molecular":
                label = batch_data["label"]
                label_mask = batch_data["label_mask"]
                if task_indices is not None:
                    if len(task_indices) != num_tasks:
                        raise ValueError(f"Expected {num_tasks} molecular task indices, got {task_indices}.")
                    label = label[:, task_indices]
                    label_mask = label_mask[:, task_indices]
            else:
                label = batch_data["subtype"].view(-1).long()
                label_mask = batch_data["subtype_mask"].view(-1).long()

        if model_engine is not None:
            cls_logit, seg_logit = model_engine(batch_data)
        else:
            cls_logit, seg_logit = model(batch_data)

        if use_seg:
            seg_pred_list = decollate_batch(seg_logit)
            seg_gt_list = decollate_batch(seg_gt)
            seg_pred_binary = [post_seg(p) for p in seg_pred_list]

            for pred, target in zip(seg_pred_binary, seg_gt_list):
                pred_bool = pred.bool()
                target_bool = target.bool()
                tp = (pred_bool & target_bool).flatten(1).sum(1)
                fp = (pred_bool & ~target_bool).flatten(1).sum(1)
                fn = (~pred_bool & target_bool).flatten(1).sum(1)
                stat: Dict[str, Any] = {"tp": tp.cpu().tolist(), "fp": fp.cpu().tolist(), "fn": fn.cpu().tolist()}
                # BraTS channel order is TC, WT, ET. An ET copy of its TC parent makes the
                # first two diagnostics approach 1 and raises parent-shell leakage.
                if pred_bool.shape[0] >= 3:
                    pred_tc, pred_et = pred_bool[0], pred_bool[2]
                    gt_tc, gt_et = target_bool[0], target_bool[2]
                    overlap = (pred_tc & pred_et).sum().item()
                    pred_tc_n = pred_tc.sum().item()
                    pred_et_n = pred_et.sum().item()
                    gt_tc_n = gt_tc.sum().item()
                    gt_et_n = gt_et.sum().item()
                    parent_shell = gt_tc & ~gt_et
                    stat.update({
                        "pred_et_tc_dice": (2.0 * overlap) / max(pred_tc_n + pred_et_n, 1),
                        "pred_et_to_tc_ratio": pred_et_n / max(pred_tc_n, 1),
                        "gt_et_to_tc_ratio": gt_et_n / max(gt_tc_n, 1),
                        "et_parent_shell_fp": (pred_et & parent_shell).sum().item(),
                        "parent_shell_voxels": parent_shell.sum().item(),
                    })
                seg_stats_local.append(stat)

            dice_metric(y_pred=seg_pred_binary, y=seg_gt_list)
            if return_outputs:
                seg_preds_list.extend([p.cpu().numpy() for p in seg_pred_binary])
                seg_labels_list.extend([g.cpu().numpy() for g in seg_gt_list])
                seg_images_list.extend([img.cpu().float().numpy() for img in decollate_batch(batch_data["image"])])
                seg_indices_list.extend(indices.detach().cpu().long().tolist())

        if use_cls:
            cls_source_dtypes.append(str(cls_logit.dtype).replace("torch.", ""))
            if cls_logit.ndim == 1:
                cls_logit = cls_logit.unsqueeze(1)
            if cls_logit.ndim != 2 or cls_logit.shape[1] != num_tasks:
                raise ValueError(f"Classification logits must have shape [B, {num_tasks}], got {tuple(cls_logit.shape)}.")

            if task_type == "molecular":
                if label.ndim == 1:
                    label = label.unsqueeze(1)
                if label_mask.ndim == 1:
                    label_mask = label_mask.unsqueeze(1)
                if cls_logit.shape != label.shape or label_mask.shape != label.shape:
                    raise ValueError(
                        f"Molecular output mismatch: logits={tuple(cls_logit.shape)}, "
                        f"labels={tuple(label.shape)}, masks={tuple(label_mask.shape)}."
                    )
                logits_fp32, probs_fp32 = _classification_probabilities_fp32(
                    cls_logit, task_type
                )
                cls_logits_fp32.extend(logits_fp32.cpu().tolist())
                cls_probs.extend(probs_fp32.cpu().tolist())
                cls_labels.extend(label.detach().cpu().long().tolist())
                cls_masks.extend(label_mask.detach().cpu().long().tolist())
            else:
                if label.shape[0] != cls_logit.shape[0] or label_mask.shape != label.shape:
                    raise ValueError(
                        f"Subtype output mismatch: logits={tuple(cls_logit.shape)}, "
                        f"labels={tuple(label.shape)}, masks={tuple(label_mask.shape)}."
                    )
                valid = label_mask == 1
                invalid = valid & ((label < 0) | (label >= num_tasks))
                if invalid.any():
                    raise ValueError(f"Valid subtype targets must be in [0, {num_tasks - 1}], got {label[invalid].tolist()}.")
                logits_fp32, probs_fp32 = _classification_probabilities_fp32(
                    cls_logit, task_type
                )
                cls_logits_fp32.extend(logits_fp32.cpu().tolist())
                cls_probs.extend(probs_fp32.cpu().tolist())
                cls_labels.extend(label.detach().cpu().long().tolist())
                cls_masks.extend(label_mask.detach().cpu().long().tolist())

        all_indices.extend(indices.cpu().tolist())

    eval_time = time.time() - start_time

    seg_dices_local: List[List[float]] = dice_metric.get_buffer().cpu().tolist() if use_seg else []

    if is_distributed():
        local_data = {
            "indices":    all_indices,
            "seg_dices":  seg_dices_local,
            "seg_stats":   seg_stats_local,
            "cls_logits": cls_logits_fp32,
            "cls_probs":  cls_probs,
            "cls_labels": cls_labels,
            "cls_masks":  cls_masks,
            "cls_source_dtypes": cls_source_dtypes,
        }

        gathered = [None] * dist.get_world_size() if is_main_process() else None
        dist.gather_object(local_data, gathered, dst=0, group=object_group)

        ## 09.16
        if not is_main_process():
            return ({}, {}, []) if return_outputs else ({}, [])

        merged_indices:    List[int] = []
        merged_seg_dices:  List[List[float]] = []
        merged_seg_stats:   List[Dict[str, Any]] = []
        merged_cls_logits: List[List[float]] = []
        merged_cls_probs:  List[List[float]] = []
        merged_cls_labels: List[List[int]] = []
        merged_cls_masks:  List[List[int]] = []
        merged_cls_source_dtypes: List[str] = []

        for item in gathered:
            merged_indices.extend(item["indices"])
            merged_seg_dices.extend(item["seg_dices"])
            merged_seg_stats.extend(item["seg_stats"])
            merged_cls_logits.extend(item["cls_logits"])
            merged_cls_probs.extend(item["cls_probs"])
            merged_cls_labels.extend(item["cls_labels"])
            merged_cls_masks.extend(item["cls_masks"])
            merged_cls_source_dtypes.extend(item["cls_source_dtypes"])

        seen = set()
        valid_mask = []
        for idx in merged_indices:
            if idx not in seen:
                seen.add(idx)
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        seg_dices_final = [d for d, v in zip(merged_seg_dices,  valid_mask) if v]
        seg_stats_final  = [s for s, v in zip(merged_seg_stats, valid_mask) if v]
        cls_indices      = [idx for idx, v in zip(merged_indices, valid_mask) if v]
        cls_logits_fp32 = [x for x, v in zip(merged_cls_logits, valid_mask) if v]
        cls_probs       = [p for p, v in zip(merged_cls_probs,  valid_mask) if v]
        cls_labels      = [l for l, v in zip(merged_cls_labels, valid_mask) if v]
        cls_masks       = [m for m, v in zip(merged_cls_masks,  valid_mask) if v]
        cls_source_dtypes = merged_cls_source_dtypes

    else:
        seg_dices_final = seg_dices_local
        seg_stats_final = seg_stats_local
        cls_indices = all_indices

    metrics: Dict[str, Dict[str, float]] = {}

    if use_seg:
        seg_dices_np = np.array(seg_dices_final)
        per_class_dice = np.nanmean(seg_dices_np, axis=0)
        seg_metrics: Dict[str, float] = {f"dice_cls{i}": float(per_class_dice[i]) for i in range(len(per_class_dice))}
        seg_metrics["dice_mean"] = float(np.nanmean(per_class_dice))
        region_names = ("TC", "WT", "ET")
        stats_tp = np.asarray([s["tp"] for s in seg_stats_final], dtype=np.float64).sum(axis=0)
        stats_fp = np.asarray([s["fp"] for s in seg_stats_final], dtype=np.float64).sum(axis=0)
        stats_fn = np.asarray([s["fn"] for s in seg_stats_final], dtype=np.float64).sum(axis=0)
        for i in range(len(per_class_dice)):
            name = region_names[i].lower() if i < len(region_names) else f"cls{i}"
            seg_metrics[f"dice_{name}"] = float(per_class_dice[i])
            seg_metrics[f"precision_{name}"] = float(stats_tp[i] / max(stats_tp[i] + stats_fp[i], 1.0))
            seg_metrics[f"recall_{name}"] = float(stats_tp[i] / max(stats_tp[i] + stats_fn[i], 1.0))
            seg_metrics[f"pred_voxels_per_case_{name}"] = float((stats_tp[i] + stats_fp[i]) / max(len(seg_stats_final), 1))
            seg_metrics[f"gt_voxels_per_case_{name}"] = float((stats_tp[i] + stats_fn[i]) / max(len(seg_stats_final), 1))
        if len(per_class_dice) >= 3:
            seg_metrics["et_tc_copy_dice"] = float(np.mean([s["pred_et_tc_dice"] for s in seg_stats_final]))
            seg_metrics["pred_et_to_tc_ratio"] = float(np.mean([s["pred_et_to_tc_ratio"] for s in seg_stats_final]))
            seg_metrics["gt_et_to_tc_ratio"] = float(np.mean([s["gt_et_to_tc_ratio"] for s in seg_stats_final]))
            shell_fp = sum(s["et_parent_shell_fp"] for s in seg_stats_final)
            shell_n = sum(s["parent_shell_voxels"] for s in seg_stats_final)
            seg_metrics["et_parent_shell_leakage"] = float(shell_fp / max(shell_n, 1))
        metrics["seg"] = seg_metrics

    if use_cls:
        if task_type == "molecular":
            metrics["cls"] = compute_multitask_binary_metrics(
                labels=cls_labels,
                probs=cls_probs,
                masks=cls_masks,
                tasks=tasks,
                threshold=cls_threshold,
            )
        else:
            metrics["cls"] = compute_multiclass_metrics(cls_labels, cls_probs, cls_masks, num_tasks)

    def _fmt(v):
        try:
            if np.isnan(v):
                return "nan"
        except TypeError:
            pass
        return f"{float(v):.4f}"

    summary_lines = []

    line = f"[Eval] total={total_samples} samples | time={eval_time:.1f}s"
    print(f"\n{line}")
    summary_lines.append(line)

    if use_seg:
        line = "  Seg | " + " | ".join(
            f"{k}: {v:.4f}" for k, v in metrics["seg"].items()
        )
        print(line)
        summary_lines.append(line)

    if use_cls:
        cls_metrics = metrics["cls"]

        if task_type == "molecular":
            macro = cls_metrics["macro_avg"]

            line = "  Cls macro | " + " | ".join(
                f"{k}: {_fmt(v)}" for k, v in macro.items()
            )
            print(line)
            summary_lines.append(line)

            for task_name, task_metric in cls_metrics["tasks"].items():
                line = (
                    f"  Cls {task_name} | n: {task_metric['n']} | auc: {_fmt(task_metric['auc'])} | "
                    f"ap: {_fmt(task_metric['ap'])} | acc: {_fmt(task_metric['acc'])} | "
                    f"bal_acc: {_fmt(task_metric['bal_acc'])} | f1: {_fmt(task_metric['f1'])}"
                )
                print(line)
                summary_lines.append(line)

        else:
            line = (
                f"  Cls subtype | n: {cls_metrics['n']} | auc_macro: {_fmt(cls_metrics['auc_macro'])} | "
                f"acc: {_fmt(cls_metrics['acc'])} | bal_acc: {_fmt(cls_metrics['bal_acc'])} | "
                f"f1_macro: {_fmt(cls_metrics['f1_macro'])}"
            )
            print(line)
            summary_lines.append(line)


    if not return_outputs:
        return metrics, summary_lines

    outputs: Dict[str, Any] = {}
    if use_seg:
        outputs.update({"seg_preds": seg_preds_list, "seg_labels": seg_labels_list, "seg_images": seg_images_list, "seg_indices": seg_indices_list})
    if use_cls:
        outputs.update({
            "indices": cls_indices,
            "cls_logits": cls_logits_fp32,
            "cls_probs": cls_probs,
            "cls_labels": cls_labels,
            "cls_masks": cls_masks,
            "tasks": tasks,
            "task_type": task_type,
            "cls_logits_dtype": "float32",
            "cls_probs_dtype": "float32",
            "cls_source_logits_dtypes": sorted(set(cls_source_dtypes)),
        })
    return metrics, outputs, summary_lines
