import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    DeleteItemsd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    Resized,
    Transposed,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    MapTransform,
)

# ── ImageNet statistics (DINOv2 pretraining) ─────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Fixed key names ───────────────────────────────────────────────────────────
_LOAD_KEYS     = ["flair", "t1", "t1ce", "t2"]
_SEG_KEY       = ["seg"]
_IMAGE_KEY     = ["image"]
_MODALITY_KEYS = ["flair", "t1", "t1ce", "t2"]   # individual keys after split
_3CH_KEYS      = ["flair_3ch", "t1_3ch", "t1ce_3ch", "t2_3ch"]
SEG_REGION_NAMES = ("TC", "WT", "ET")

TARGET_SPACING = (1.0, 1.0, 1.0)
COMMON_GRID_SIZE = (240, 240, 155)

# ════════════════════════════════════════════════════════════════════════════
# Custom transforms
# ════════════════════════════════════════════════════════════════════════════

class SplitModalitiesd(MapTransform):
    """
    Split a concatenated (4, D, H, W) image tensor back into 4 individual
    (1, D, H, W) tensors stored under separate keys.

    Called after augmentation on the fused image so that all modalities
    received identical spatial transforms (guaranteed by operating on the
    same tensor), then split for independent per-modality processing.

    Input key  : "image"  shape (4, D, H, W)
    Output keys: "flair" / "t1" / "t1ce" / "t2"  each (1, D, H, W)
    """
    def __init__(self, image_key="image",
                 out_keys=("flair", "t1", "t1ce", "t2")):
        super().__init__(keys=[image_key], allow_missing_keys=False)
        self.image_key = image_key
        self.out_keys  = out_keys

    def __call__(self, data):
        d   = dict(data)
        img = d[self.image_key]               # (4, D, H, W)
        for i, key in enumerate(self.out_keys):
            d[key] = img[i : i + 1]           # (1, D, H, W) — keeps channel dim
        return d


class ValidateBratsRegionsd(MapTransform):
    """Fail fast if TC/WT/ET channel order or nesting is corrupted."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]
            if seg.ndim != 4 or seg.shape[0] != len(SEG_REGION_NAMES):
                raise ValueError(
                    f"{key} must be [3, H, W, D] in TC/WT/ET order, got {tuple(seg.shape)}."
                )
            binary = seg.bool() if isinstance(seg, torch.Tensor) else seg.astype(bool)

            def any_value(value):
                result = value.any()
                return bool(result.item() if isinstance(result, torch.Tensor) else result)

            if any_value(binary[0] & ~binary[1]):
                raise ValueError("Invalid BraTS target: TC must be contained in WT.")
            if any_value(binary[2] & ~binary[0]):
                raise ValueError("Invalid BraTS target: ET must be contained in TC.")
        return d


class RepeatChannelTo3d(MapTransform):
    """
    Repeat a single-channel tensor 3 times along dim-0 to produce a
    (3, D, H, W) tensor that DINOv2's patch-embed layer accepts.

    Input  : (1, D, H, W)  stored at key
    Output : (3, D, H, W)  stored at key + suffix  (default "_3ch")
    """
    def __init__(self, keys, suffix="_3ch"):
        super().__init__(keys=keys, allow_missing_keys=False)
        self.suffix = suffix

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key + self.suffix] = d[key].repeat(3, 1, 1, 1)  # (3, D, H, W)
        return d

class ImageNetNormalized(MapTransform):
    """
    Align MRI value range to ImageNet statistics expected by DINOv3.

    Steps applied to each (3, D, H, W) tensor:
      1. Clamp to [clip_min, clip_max]
         After NormalizeIntensity, MRI values are ~N(0,1); ±5 σ keeps >99.9%.
      2. Linear map [clip_min, clip_max] → [0, 1]
      3. Subtract ImageNet channel mean and divide by std
         (same operation as torchvision.transforms.Normalize for 2-D images,
          applied along the channel dim with broadcasting over D, H, W)
    """
    def __init__(self, keys,
                 mean=IMAGENET_MEAN,
                 std=IMAGENET_STD,
                 clip_range=(-5.0, 5.0)):
        super().__init__(keys=keys, allow_missing_keys=False)
        self.mean      = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std       = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1, 1)
        self.clip_min  = clip_range[0]
        self.clip_span = clip_range[1] - clip_range[0]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img  = d[key].float()
            img  = img.clamp(self.clip_min, self.clip_min + self.clip_span)
            img  = (img - self.clip_min) / self.clip_span   # → [0, 1]
            mean = self.mean.to(img.device)
            std  = self.std.to(img.device)
            d[key] = (img - mean) / std
        return d

def make_train_transform(
    *,
    spatial_size=(224, 224, 32),
    # spatial_size=(224, 224, 128),
    hflip_prob:  float = 0.0,
    rot90_prob:  float = 0.3,
    scale_prob:  float = 0.5,
    shift_prob:  float = 0.5,
    noise_prob:  float = 0.2,
    noise_std:   float = 0.01,
) -> Compose:
    """
    Training transform.

    Spatial augmentation (flip, rotate) acts on the fused (4, D, H, W) image
    together with seg, guaranteeing all modalities and the mask receive
    identical deformations.  Intensity augmentation is applied per-channel
    so each modality is jittered independently (models scanner variability).

    Final output shapes  (D=32, H=224, W=224):
      flair_3ch / t1_3ch / t1ce_3ch / t2_3ch : (3, 32, 224, 224)
      seg                                      : (3, 32, 224, 224)
    """
    return Compose([
        # ── 1. Load ────────────────────────────────────────────────────────
        LoadImaged(keys=_LOAD_KEYS + _SEG_KEY, reader="NibabelReader", ensure_channel_first=True, image_only=True),
        Orientationd(keys=_LOAD_KEYS + _SEG_KEY, axcodes="RAS"),

        # ── 2. Convert seg {0,1,2,4} → 3-channel binary (TC / WT / ET) ────
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
        ValidateBratsRegionsd(keys="seg"),

        # ── 3. Concatenate 4 modalities → (4, H, W, D) ────────────────────
        ConcatItemsd(keys=_LOAD_KEYS, name="image", dim=0),

        # ── 4. Crop away air background before resize ──────────────────────
        CropForegroundd(keys=_IMAGE_KEY + _SEG_KEY, source_key="image", select_fn=lambda x: x > 0, margin=5,),
        DeleteItemsd(keys=["foreground_start_coord", "foreground_end_coord"]),

        # ── 5. MRI intensity normalisation ────────────────────────────────
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

        # ── 6. Resize to (H=224, W=224, D=32) ─────────────────────────────
        Resized(keys=_IMAGE_KEY, spatial_size=spatial_size),
        Resized(keys=_SEG_KEY,   spatial_size=spatial_size, mode="nearest"),

        # ── 7. Reorder axes (C, H, W, D) → (C, D, H, W) ──────────────────
        Transposed(keys=_IMAGE_KEY + _SEG_KEY, indices=[0, 3, 1, 2]),

        # ── 8. Spatial augmentation (on fused image + seg together) ────────
        RandFlipd(keys=_IMAGE_KEY + _SEG_KEY, prob=hflip_prob, spatial_axis=-1),
        RandRotate90d(keys=_IMAGE_KEY + _SEG_KEY, prob=rot90_prob, max_k=3, spatial_axes=(1, 2)),

        # ── 9. Intensity augmentation (image only, not seg) ────────────────
        RandScaleIntensityd(keys="image", factors=0.1, prob=scale_prob, channel_wise=True),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=shift_prob, channel_wise=True),
        RandGaussianNoised(keys="image", prob=noise_prob, mean=0.0, std=noise_std),

        # ── 10. Convert to tensors ─────────────────────────────────────────
        ToTensord(keys=_IMAGE_KEY + _SEG_KEY + ["label", "label_mask", "subtype", "subtype_mask"], track_meta=False, allow_missing_keys=True),

        # ── 11. Split (4, D, H, W) → 4 × (1, D, H, W) ────────────────────
        SplitModalitiesd(image_key="image", out_keys=("flair", "t1", "t1ce", "t2")),

        # ── 12. Repeat channel × 3: (1, D, H, W) → (3, D, H, W) ──────────
        RepeatChannelTo3d(keys=_MODALITY_KEYS, suffix="_3ch"),

        # ── 13. ImageNet normalisation ─────────────────────────────────────
        ImageNetNormalized(keys=_3CH_KEYS),

        DeleteItemsd(keys=_MODALITY_KEYS)
    ])


def make_eval_transform(
    *,
    spatial_size=(224, 224, 32),
) -> Compose:
    """
    Deterministic eval/test transform.
    Identical to make_train_transform minus steps 8 and 9 (all augmentation).
    Output shapes are the same.
    """
    return Compose([
        LoadImaged(keys=_LOAD_KEYS + _SEG_KEY, reader="NibabelReader", ensure_channel_first=True, image_only=True),
        Orientationd(keys=_LOAD_KEYS + _SEG_KEY, axcodes="RAS"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
        ValidateBratsRegionsd(keys="seg"),
        ConcatItemsd(keys=_LOAD_KEYS, name="image", dim=0),
        CropForegroundd(keys=_IMAGE_KEY + _SEG_KEY, source_key="image", select_fn=lambda x: x > 0, margin=5,),
        DeleteItemsd(keys=["foreground_start_coord", "foreground_end_coord"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=_IMAGE_KEY, spatial_size=spatial_size),
        Resized(keys=_SEG_KEY,   spatial_size=spatial_size, mode="nearest"),
        Transposed(keys=_IMAGE_KEY + _SEG_KEY, indices=[0, 3, 1, 2]),
        ToTensord(keys=_IMAGE_KEY + _SEG_KEY + ["label", "label_mask", "subtype", "subtype_mask"], track_meta=False, allow_missing_keys=True),
        SplitModalitiesd(image_key="image", out_keys=("flair", "t1", "t1ce", "t2")),
        RepeatChannelTo3d(keys=_MODALITY_KEYS, suffix="_3ch"),
        ImageNetNormalized(keys=_3CH_KEYS),
        DeleteItemsd(keys=_MODALITY_KEYS)
    ])
