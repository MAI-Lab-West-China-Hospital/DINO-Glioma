import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

try:
    import torch.distributed as dist
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()
 
 
def is_main_process() -> bool:
    if is_distributed():
        return dist.get_rank() == 0
    return True

def format_metric(value: Any) -> str:
    """Format scalar metric safely."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"

    if math.isnan(value):
        return "nan"

    return f"{value:.4f}"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_state_dict(
    epoch: int,
    feature_model: nn.Module,
    classifier_head: Optional[nn.Module],
    segment_head: Optional[nn.Module],
    extra: Optional[Dict] = None,
) -> Dict:
    state = {
        "epoch": epoch,
        "feature_model": feature_model.state_dict(),
    }
    if classifier_head is not None:
        state["classifier_head"] = classifier_head.state_dict()
    if segment_head is not None:
        state["segment_head"] = segment_head.state_dict()
    if extra:
        state.update(extra)
    return state


def save_checkpoint(
    path: Path,
    epoch: int,
    feature_model: nn.Module,
    classifier_head: Optional[nn.Module],
    segment_head: Optional[nn.Module],
    metrics: Dict,
    model_engine: Optional[Any],
    ckpt_dir: Optional[Path],
    tag: str,
    logger: Optional[logging.Logger] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save checkpoint in the correct format for DeepSpeed or plain PyTorch.
    """
    if model_engine is not None:
        # DeepSpeed: all ranks must call save_checkpoint together.
        client_state = {
            "epoch": epoch,
            "metrics": metrics,
            "tag": tag,
            **(checkpoint_metadata or {}),
        }
        model_engine.save_checkpoint(
            str(ckpt_dir),
            tag=tag,
            client_state=client_state,
        )
        if logger and is_main_process():
            logger.info(f"[Ckpt] DeepSpeed checkpoint saved: dir={ckpt_dir}, tag={tag}")

    else:
        # Plain PyTorch: rank 0 only.
        path.parent.mkdir(parents=True, exist_ok=True)
        state = _build_state_dict(
            epoch=epoch,
            feature_model=feature_model,
            classifier_head=classifier_head,
            segment_head=segment_head,
            extra={"metrics": metrics, **(checkpoint_metadata or {})},
        )
        torch.save(state, path)

        if logger:
            logger.info(f"[Ckpt] Saved checkpoint → {path}")


def _safe_metric_value(value: Any, default: float = float("nan")) -> float:
    """Convert one metric value to float safely."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def _get_cls_score_from_multitask_metrics(
    cls_metrics: Dict,
    primary_key: str = "auc",
    fallback_key: str = "bal_acc",
    average_key: str = "macro_avg",
) -> float:
    """
    Get one scalar classification score from multi-task classification metrics.

    Default:
        use cls["macro_avg"]["auc"];
        if AUC is NaN, fall back to cls["macro_avg"]["bal_acc"].
    """
    avg_metrics = cls_metrics.get(average_key, {})

    primary = _safe_metric_value(avg_metrics.get(primary_key, float("nan")))
    fallback = _safe_metric_value(avg_metrics.get(fallback_key, float("nan")))

    if not math.isnan(primary):
        return primary

    if not math.isnan(fallback):
        return fallback

    return float("nan")


def update_best_checkpoints(
    epoch: int,
    val_metrics: Dict[str, Dict[str, Any]],
    best_scores: Dict[str, float],
    feature_model: nn.Module,
    classifier_head: Optional[nn.Module],
    segment_head: Optional[nn.Module],
    ckpt_dir: Path,
    model_engine: Optional[Any],
    logger: Optional[logging.Logger] = None,
    task_type: str = "molecular",
    use_cls: bool = True,
    use_seg: bool = True,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """
    Evaluate checkpoint criteria and save if improved.

    Criteria
    --------
    cls:
        molecular uses macro-average AUC with balanced-accuracy fallback;
        subtype uses macro OVR AUC with balanced-accuracy fallback.

    seg:
        mean Dice.

    joint:
        cls_score + seg_score.
    """
    cls_m = val_metrics.get("cls", {})
    seg_m = val_metrics.get("seg", {})

    if use_cls:
        if task_type == "molecular":
            cls_score = _get_cls_score_from_multitask_metrics(
                cls_metrics=cls_m, primary_key="auc", fallback_key="bal_acc", average_key="macro_avg",
            )
        elif task_type == "subtype":
            primary = _safe_metric_value(cls_m.get("auc_macro", float("nan")))
            fallback = _safe_metric_value(cls_m.get("bal_acc", float("nan")))
            # cls_score = primary if not math.isnan(primary) else fallback
            cls_score = 0.5 * primary + 0.5 * fallback            
        else:
            raise ValueError(f"Unknown task_type '{task_type}'.")

    if use_seg:
        mean_dice = _safe_metric_value(seg_m.get("dice_mean", float("nan")))
        seg_score = mean_dice if not math.isnan(mean_dice) else -math.inf

    improved: Dict[str, bool] = {
        "cls": False,
        "seg": False,
        "joint": False,
    }

    criteria = {}
    if use_cls:
        criteria["cls"] = (cls_score, "best_cls_model.pth", "best_cls")
    if use_seg:
        criteria["seg"] = (seg_score, "best_seg_model.pth", "best_seg")
    if use_cls and use_seg:
        joint_score = (
            (cls_score if not math.isnan(cls_score) else 0.0)
            + (seg_score if seg_score != -math.inf else 0.0)
        )
        criteria["joint"] = (joint_score, "best_joint_model.pth", "best_joint")

    for name, (score, fname, ds_tag) in criteria.items():
        if math.isnan(score) or score == -math.inf:
            continue

        if score > best_scores[name]:
            best_scores[name] = score
            improved[name] = True

            if model_engine is not None or is_main_process():
                save_checkpoint(
                    path=ckpt_dir / fname,
                    epoch=epoch,
                    feature_model=feature_model,
                    classifier_head=classifier_head,
                    segment_head=segment_head,
                    metrics=val_metrics,
                    model_engine=model_engine,
                    ckpt_dir=ckpt_dir,
                    tag=ds_tag,
                    logger=logger,
                    checkpoint_metadata=checkpoint_metadata,
                )

            if logger and is_main_process():
                logger.info(
                    f"[Ckpt] New best [{name}] score={score:.4f} at epoch {epoch}"
                )

    return improved
