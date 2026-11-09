import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def compute_multitask_binary_metrics(
    labels,
    probs,
    masks,
    tasks,
    threshold: Optional[Union[float, Dict[str, float]]] = 0.5,
):

    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs, dtype=float)
    masks_np = np.asarray(masks)

    if labels_np.ndim == 1:
        labels_np = labels_np[:, None]
    if probs_np.ndim == 1:
        probs_np = probs_np[:, None]
    if masks_np.ndim == 1:
        masks_np = masks_np[:, None]

    if labels_np.shape != probs_np.shape or masks_np.shape != labels_np.shape:
        raise ValueError(
            f"Classification metric shape mismatch: "
            f"labels={labels_np.shape}, probs={probs_np.shape}, masks={masks_np.shape}."
        )

    task_metrics = {}
    metric_names = ["auc", "ap", "acc", "bal_acc", "precision", "recall", "specificity", "f1"]  # [A1]

    for task_idx, task_name in enumerate(tasks):
        valid = masks_np[:, task_idx].astype(bool)
        y_true = labels_np[valid, task_idx].astype(int)
        y_prob = probs_np[valid, task_idx]

        n = int(valid.sum())
        missing = int((~valid).sum())
        positive = int((y_true == 1).sum()) if n > 0 else 0
        negative = int((y_true == 0).sum()) if n > 0 else 0

        if n == 0:
            task_metrics[task_name] = {
                "n": n,
                "positive": positive,
                "negative": negative,
                "missing": missing,
                **{k: float("nan") for k in metric_names},
            }
            continue

        auc = (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )
        ap = (  # [A1]
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )

        if threshold is None: 
            acc = float("nan")
            bal_acc = float("nan")
            precision = float("nan")
            recall = float("nan")
            specificity = float("nan")
            f1 = float("nan")
        else:
            task_threshold = (
                float(threshold.get(task_name, 0.5))
                if isinstance(threshold, dict)
                else float(threshold)
            )
            y_pred = (y_prob >= task_threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            recall = recall_score(y_true, y_pred, zero_division=0)

            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))  # [A3]
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            specificity = float(specificity)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))

        task_metrics[task_name] = {
            "n": n,
            "positive": positive,
            "negative": negative,
            "missing": missing,
            "auc": auc,
            "ap": ap, 
            "acc": acc,
            "bal_acc": bal_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        }

    macro_avg = {}
    for name in metric_names:
        values = np.asarray([task_metrics[t][name] for t in tasks], dtype=float)
        macro_avg[name] = (
            float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")
        ) 
    valid_all = masks_np.astype(bool)
    y_true_all = labels_np[valid_all].astype(int)
    y_prob_all = probs_np[valid_all]

    if len(y_true_all) == 0:
        micro_avg = {
            "auc": float("nan"),
            "ap": float("nan"),  
            "acc": float("nan"),
            "bal_acc": float("nan"), 
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
        }
    else:
        micro_auc = (
            float(roc_auc_score(y_true_all, y_prob_all))
            if len(np.unique(y_true_all)) == 2
            else float("nan")
        )
        micro_ap = (  # [A1]
            float(average_precision_score(y_true_all, y_prob_all))
            if len(np.unique(y_true_all)) == 2
            else float("nan")
        )

        if threshold is None or isinstance(threshold, dict):  
            micro_avg = {
                "auc": micro_auc,
                "ap": micro_ap,  
                "acc": float("nan"),
                "bal_acc": float("nan"), 
                "precision": float("nan"),
                "recall": float("nan"),
                "specificity": float("nan"),
                "f1": float("nan"),
            }
        else:
            y_pred_all = (y_prob_all >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

            micro_avg = {
                "auc": micro_auc,
                "ap": micro_ap,  
                "acc": float(accuracy_score(y_true_all, y_pred_all)),
                "bal_acc": float(balanced_accuracy_score(y_true_all, y_pred_all)), 
                "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
                "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
                "specificity": float(specificity),
                "f1": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
            }

    return {
        "tasks": task_metrics,
        "macro_avg": macro_avg,
        "micro_avg": micro_avg,
    }

def compute_multiclass_metrics(labels, probs, masks, num_classes):
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=float)
    masks = np.asarray(masks, dtype=int).reshape(-1)
    if probs.ndim != 2 or probs.shape != (labels.size, num_classes) or masks.size != labels.size:
        raise ValueError(f"Multiclass output mismatch: labels={labels.shape}, probs={probs.shape}, masks={masks.shape}.")
    valid = masks == 1
    y_true, y_prob = labels[valid], probs[valid]
    if np.any((y_true < 0) | (y_true >= num_classes)):
        raise ValueError(f"Valid subtype targets must be in [0, {num_classes - 1}].")
    classes = list(range(num_classes))
    if y_true.size == 0:
        return {
            "n": 0, "missing": int(labels.size), "auc_macro": float("nan"), "acc": float("nan"),
            "bal_acc": float("nan"), "f1_macro": float("nan"),
            "per_class": {str(i): {"support": 0, "sensitivity": float("nan")} for i in classes},
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
        }
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    auc_macro = float("nan")
    auc_per_class = np.full(num_classes, np.nan, dtype=float)

    if len(np.unique(y_true)) == num_classes:
        try:
            auc_per_class = roc_auc_score(
                y_true,
                y_prob,
                labels=classes,
                multi_class="ovr",
                average=None,
            )
            auc_macro = float(np.mean(auc_per_class))
        except ValueError:
            pass
    per_class = {}
    for i in classes:
        support = int(cm[i].sum())
        per_class[str(i)] = {
            "support": support,
            "sensitivity": float(cm[i, i] / support) if support else float("nan"),
            "auc_ovr": float(auc_per_class[i]),
        }
    return {
        "n": int(y_true.size), "missing": int(labels.size - y_true.size), "auc_macro": auc_macro,
        "acc": float(accuracy_score(y_true, y_pred)), "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "per_class": per_class, "confusion_matrix": cm.tolist(),
    }
