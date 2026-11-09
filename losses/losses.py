import torch
from torch import nn
from typing import Callable, Optional, Sequence
import torch.nn.functional as F

def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)

class WeightedTverskyLoss(nn.Module):
    """Region-wise Tversky loss for overlapping sigmoid targets.

    ``alpha`` weights false positives and ``beta`` weights false negatives.
    Per-channel values are supported so a small child region can receive a
    stronger false-positive penalty than its parent region.
    """

    def __init__(
        self,
        apply_nonlin: Optional[Callable] = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1e-5,
        ddp: bool = False,
        alpha: Sequence[float] = (0.5, 0.5, 0.7),
        beta: Sequence[float] = (0.5, 0.5, 0.3),
        class_weights: Sequence[float] = (1.0, 1.0, 2.0),
        reduction: str = "none"
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.reduction = reduction
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float32))
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    @staticmethod
    def _channel_vector(values: torch.Tensor, channels: int, reference: torch.Tensor) -> torch.Tensor:
        if values.numel() == 1:
            values = values.expand(channels)
        if values.numel() != channels:
            raise ValueError(f"Expected {channels} per-class values, got {values.numel()}.")
        return values.to(device=reference.device, dtype=reference.dtype)

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None) -> torch.Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if x.shape != y.shape:
            raise ValueError(
                "WeightedTverskyLoss expects overlapping one-hot/region targets with "
                f"the same shape as logits, got logits={tuple(x.shape)}, target={tuple(y.shape)}."
            )

        y = y.to(dtype=x.dtype)
        if not self.do_bg:
            x, y = x[:, 1:], y[:, 1:]

        if loss_mask is not None:
            mask = loss_mask.to(dtype=x.dtype).expand_as(x)
            x, y = x * mask, y * mask

        spatial_axes = tuple(range(2, x.ndim))
        axes = (0, *spatial_axes) if self.batch_dice else spatial_axes
        tp = (x * y).sum(axes)
        fp = (x * (1.0 - y)).sum(axes)
        fn = ((1.0 - x) * y).sum(axes)

        channels = x.shape[1]
        alpha_values = self.alpha[1:] if not self.do_bg and self.alpha.numel() > 1 else self.alpha
        beta_values = self.beta[1:] if not self.do_bg and self.beta.numel() > 1 else self.beta
        weight_values = self.class_weights[1:] if not self.do_bg and self.class_weights.numel() > 1 else self.class_weights
        alpha = self._channel_vector(alpha_values, channels, x)
        beta = self._channel_vector(beta_values, channels, x)
        weights = self._channel_vector(weight_values, channels, x)
        if not self.batch_dice:
            alpha, beta, weights = alpha[None], beta[None], weights[None]

        score = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        weighted = (1.0 - score) * weights
        if self.batch_dice:
            loss = weighted.sum() / weights.sum().clamp_min(1e-8)
            return loss
        # weighted: [B, C]
        loss_per_sample = (weighted.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8))
        if self.reduction == "none":
            return loss_per_sample
        return loss_per_sample.mean()

class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
        if not self.do_bg:
            x = x[:, 1:]

        y_onehot = y_onehot.to(dtype=x.dtype)
        if loss_mask is not None:
            mask = loss_mask.to(dtype=x.dtype).expand_as(x)
            x = x * mask
            y_onehot = y_onehot * mask

        spatial_axes = tuple(range(2, x.ndim))
        axes = (0, *spatial_axes) if self.batch_dice else spatial_axes
        intersect = (x * y_onehot).sum(axes)
        sum_pred = x.sum(axes)
        sum_gt = y_onehot.sum(axes)
        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        return 1.0 - dc.mean()

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, class_weights: Optional[Sequence[float]] = None, reduction: str = "none"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        bce_kwargs = dict(bce_kwargs)
        bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        self.reduction =reduction
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0))
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_per_sample = self.dc(net_output, target_regions, loss_mask=mask)
        if self.reduction == "none" and (dc_per_sample.ndim != 1 or dc_per_sample.shape[0] != net_output.shape[0]):
            raise RuntimeError(f"Expected Dice/Tversky per-sample loss [B], got {tuple(dc_per_sample.shape)} for batch size {net_output.shape[0]}.")
        target_regions = target_regions.float()
        raw_ce = self.ce(net_output, target_regions)
        spatial_dims = tuple(range(2, raw_ce.ndim))
        if mask is not None:
            valid = mask.expand_as(raw_ce)
            raw_ce = raw_ce * valid
            denom = valid.sum(dim=spatial_dims).clamp_min(1e-8)
            ce_per_sample_class = (raw_ce.sum(dim=spatial_dims) / denom)
        else:
            ce_per_sample_class = raw_ce.mean(dim=spatial_dims)

        if self.class_weights.numel():
            if self.class_weights.numel() != raw_ce.shape[1]:
                raise ValueError(f"Expected {raw_ce.shape[1]} BCE class weights, got {self.class_weights.numel()}.")
            weights = self.class_weights.to(device=raw_ce.device, dtype=raw_ce.dtype)
            ce_per_sample = (ce_per_sample_class * weights[None, :]).sum(dim=1) / weights.sum().clamp_min(1e-8)
        else:
            ce_per_sample = ce_per_sample_class.mean(dim=1)
        # [B]
        result_per_sample = self.weight_ce * ce_per_sample + self.weight_dice * dc_per_sample
        if self.reduction == "none":
            return result_per_sample
        return result_per_sample.mean()

def classification_statistics(records, tasks, task_indices, num_classes=2):
    """Count the complete, uniformly sampled training pool before sharding.

    Molecular records use task_indices; subtype records use task_indices=None.
    Masks are authoritative, including dataset-disabled labels.
    """
    counts = [[0] * num_classes for _ in tasks]
    for record_index, record in enumerate(records):
        labels = ([record["subtype"]] if task_indices is None
                  else [record["label"][index] for index in task_indices])
        masks = ([record["subtype_mask"]] if task_indices is None
                 else [record["label_mask"][index] for index in task_indices])
        for task_index, (label, mask) in enumerate(zip(labels, masks)):
            if mask not in (0, 1):
                raise ValueError(f"Training record {record_index}, task {tasks[task_index]}: mask must be 0 or 1.")
            if mask == 0:
                continue
            if label not in range(num_classes):
                raise ValueError(f"Training record {record_index}, task {tasks[task_index]}: observed label {label!r} must be an integer in [0, {num_classes - 1}].")
            counts[task_index][int(label)] += 1

    valid = [sum(task_counts) for task_counts in counts]
    for task, task_counts in zip(tasks, counts):
        absent = [index for index, count in enumerate(task_counts) if count == 0]
        if absent:
            raise ValueError(f"Active classification task {task!r} has no observed training records for class(es) {absent}. Add labeled training records, correct masks/split, or disable this task.")
    return {
        "sampling_unit": "record",
        "normalization": "sum(mask * class_weight * loss / availability) / (batch_size * task_count)",
        "tasks": list(tasks),
        "task_indices": None if task_indices is None else list(task_indices),
        "num_records": len(records),
        "class_counts": counts,
        "valid_counts": valid,
        "missing_counts": [len(records) - count for count in valid],
        "class_weights": [[count / (num_classes * n) for n in task_counts]
                          for count, task_counts in zip(valid, counts)],
        "availability": [count / len(records) for count in valid],
    }


def masked_bce_loss(logits, labels, label_mask, class_weights, availability):
    """BCE estimator with fixed training availability; logits/targets are [B,T]."""
    targets = torch.where(label_mask.bool(), labels, 0).float()
    raw_loss = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    weights = torch.where(targets == 1, class_weights[:, 1], class_weights[:, 0])
    return (raw_loss * label_mask * weights / availability).mean()


def masked_ce_loss(logits, labels, label_mask, class_weights, availability):
    """Hard-label CE estimator, including unlabeled records in the batch mean."""
    targets = torch.where(label_mask.bool(), labels, 0).long()
    raw_loss = F.cross_entropy(logits.float(), targets, weight=class_weights[0], reduction="none")
    return (raw_loss * label_mask / availability[0]).mean()

