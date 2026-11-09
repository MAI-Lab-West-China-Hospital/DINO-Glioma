from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .encoder_decoder import UNETRDecoder3D

from enum import Enum
import torch

class BackboneLayersSet(Enum):
    """
    Set of intermediate layers to take from the backbone.
    """

    LAST = "LAST"  # extracting only the last layer
    FOUR_LAST = "FOUR_LAST"  # extracting the four last layers
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"  # extracting outputs every 1/4 of the total number of blocks


def get_backbone_out_indices(
    model: torch.nn.Module,
    backbone_out_layers: BackboneLayersSet = BackboneLayersSet.FOUR_EVEN_INTERVALS,
):
    """
    Get indices for output layers of the ViT backbone. For now there are 3 options available:
    BackboneLayersSet.LAST : only extract the last layer, used in segmentation tasks with a bn head.
    BackboneLayersSet.FOUR_EVEN_INTERVALS : extract outputs every 1/4 of the total number of blocks
    Reference outputs in 'FOUR_EVEN_INTERVALS' mode :
    ViT/S (12 blocks): [2, 5, 8, 11]
    ViT/B (12 blocks): [2, 5, 8, 11]
    ViT/L (24 blocks): [5, 11, 17, 23] (classic), [4, 11, 17, 23] (used in the paper)
    ViT/g (40 blocks): [9, 19, 29, 39]
    """
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        # Take indices that were used in the paper (for ViT/L only)
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    assert all([out_index < n_blocks for out_index in out_indices])
    return out_indices

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _infer_dim(sample_output, use_n_blocks, use_patch_token, modal_fusion):
    modality_keys = list(sample_output.keys())
    num_modalities = len(modality_keys)
    first_mod = modality_keys[0]
    intermediate = sample_output[first_mod][-use_n_blocks:]
    cls_dim = sum(cls.shape[-1] for _, cls in intermediate) # if use_n_blocks is 4 ==4*dim, if one then 1*dim
    if use_patch_token:
        patch_dim = intermediate[-1][0].shape[-1]
        single_mod_dim = cls_dim + patch_dim
    else:
        single_mod_dim = cls_dim
    if modal_fusion == "concat":
        out_dim = single_mod_dim*num_modalities
    elif modal_fusion in ("mean", "attention"):
        out_dim = single_mod_dim
    else:
        raise ValueError(f"Unknown modality_fusion:{modal_fusion}")
    return modality_keys, out_dim

# ---------------------------------------------------------------------------
# 1. Per-modality extractor
# ---------------------------------------------------------------------------

class ModalityExtractor(nn.Module):
    """
    Selects the last `use_n_blocks` blocks from one modality and produces

    Feature construction:
        cls       = cat(c_i for i in last N blocks)    (B, D, N*c_dim)
        patch_avg = last_block_p.mean(dim=2)            (B, D, p_dim)  [if use_patch_token]
        x         = cat([cls, patch_avg])  or  cls
        #out       = Linear(x)                           (B, D, hidden_dim) ## no linear
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        use_n_blocks: int,
        use_patch_token: bool,
    ):
        super().__init__()
        self.use_n_blocks = use_n_blocks
        self.use_patch_token = use_patch_token

        # in_dim = use_n_blocks * c_dim + (p_dim if use_patch_token else 0)
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(
        self,
        blocks: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        # Take last N blocks
        selected = blocks[-self.use_n_blocks:]            

        # Concatenate cls tokens across selected blocks
        cls = torch.cat([c for _, c in selected], dim=-1)  
        if self.use_patch_token:
            patch_avg = selected[-1][0].mean(dim=2)         
            x = torch.cat([cls, patch_avg], dim=-1)        
        else:
            x = cls                                         

        return self.proj(x)                                 
    
# ---------------------------------------------------------------------------
# 2. Cross-modal fusion  —  operates slice-by-slice, D is never compressed
# ---------------------------------------------------------------------------

class ModalFusion(nn.Module):
    """
    Fuses M modality features at each slice position.

    Input  : list of M tensors, each (B, D, hidden_dim)
    Output : (B, D, hidden_dim)

    mode="mean"      average over modalities
    mode="concat"    cat -> Linear(M*H -> H)
    mode="gated"     softmax-gated weighted sum (gate computed per modality per slice)
    mode="attention" M modalities as tokens, TransformerEncoder, then mean-pool
    """

    def __init__(self, hidden_dim: int, num_modalities: int, mode: str):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            self.proj = nn.Linear(num_modalities * hidden_dim, hidden_dim)

        elif mode == "gated":
            self.gate = nn.Linear(hidden_dim, 1)

        elif mode == "attention":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, hidden_dim // 64),
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                norm_first=True,
            )
            self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        elif mode != "mean":
            raise ValueError(f"Unknown modal_fusion: {mode!r}")

    def forward(self, mod_feats: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(mod_feats, dim=2)       
        B, D, M, H = stacked.shape

        if self.mode == "mean":
            return stacked.mean(dim=2)                

        elif self.mode == "concat":
            return self.proj(stacked.reshape(B, D, M * H)) 

        elif self.mode == "gated":
            gates = torch.softmax(self.gate(stacked), dim=2)  
            return (gates * stacked).sum(dim=2)              

        else:  # attention
            x = stacked.reshape(B * D, M, H)          
            x = self.attn_encoder(x)                   
            return x.mean(dim=1).reshape(B, D, H)       


# ---------------------------------------------------------------------------
# 3. Slice pooling  (B, D, H) -> (B, H)
# ---------------------------------------------------------------------------

class SlicePooling(nn.Module):
    """
    Aggregates the D (slice) dimension into a single patient-level vector.

    mode="mean"      average over D
    mode="max"       max over D
    mode="attention" learned per-slice weight (softmax), weighted sum
    """

    def __init__(self, hidden_dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "attention":
            self.score = nn.Linear(hidden_dim, 1)
        elif mode not in ("mean", "max"):
            raise ValueError(f"Unknown slice_pooling: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H)
        if self.mode == "mean":
            return x.mean(dim=1)                    

        elif self.mode == "max":
            return x.max(dim=1).values                 

        else:  # attention
            w = torch.softmax(self.score(x), dim=1)   
            return (w * x).sum(dim=1)                     

# ---------------------------------------------------------------------------
# 4. MLP classifier
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LinearClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# 5. Top-level classification head
# ---------------------------------------------------------------------------

class MultimodalMRIClassHead(nn.Module):
    """
    Multimodal 3D MRI Classification Head.

    c_dim and p_dim are inferred automatically from `sample_F_all`.

    Args:
        sample_F_all    one example F_all dict (tensors can be on any device/dtype;
                        only .shape[-1] is read, no computation is performed)
        # hidden_dim      projected feature dimension used throughout the network
        num_classes     number of output classes
        use_n_blocks    how many of the *last* blocks to use per modality  (default: 1)
        use_patch_token append mean-pooled patch token from the last selected block
        modal_fusion    "mean" | "concat" | "gated" | "attention"
        slice_pooling   "mean" | "max" | "attention"
        dropout         dropout probability in the MLP classifier

    Usage:
        head = MultimodalMRIClassHead(sample_F_all, hidden_dim=512, num_classes=3,
                                      use_n_blocks=3, use_patch_token=True,
                                      modal_fusion="attention", slice_pooling="attention")
        logits = head(F_all)   # (B, num_classes)
    """

    def __init__(
        self,
        sample_F_all: Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        hidden_dim: int,
        num_classes: int = 2,
        use_n_blocks: int = 1,
        use_patch_token: bool = False,
        modal_fusion: str = "attention",
        slice_pooling: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()

        modality_keys, sample_dim = _infer_dim(sample_F_all, use_n_blocks, use_patch_token, modal_fusion)
        self.modality_keys = modality_keys

        self.extractors = nn.ModuleDict({
            key: ModalityExtractor(
                in_dim=sample_dim,
                hidden_dim=hidden_dim,
                use_n_blocks=use_n_blocks,
                use_patch_token=use_patch_token,
            )
            for key in modality_keys
        })

        self.modal_fusion = ModalFusion(
            hidden_dim=hidden_dim,
            num_modalities=len(modality_keys),
            mode=modal_fusion,
        )
        self.slice_pooling = SlicePooling(hidden_dim=hidden_dim, mode=slice_pooling)
        # self.classifier = MLPClassifier(
        #     hidden_dim=hidden_dim,
        #     num_classes=num_classes,
        #     dropout=dropout,
        # )
        self.classifier = LinearClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(
        self,
        F_all: Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> torch.Tensor:
        """
        Args:
            F_all   modality_key -> tuple of (p, c) pairs ordered by block index
        Returns:
            logits  (B, num_classes)
        """
        # Step 1: per-modality extraction -> each (B, D, hidden_dim)
        mod_feats = [self.extractors[key](F_all[key]) for key in self.modality_keys]

        # Step 2: cross-modal fusion per slice -> (B, D, hidden_dim)
        slice_feat = self.modal_fusion(mod_feats)

        # Step 3: slice pooling -> (B, hidden_dim)
        patient_feat = self.slice_pooling(slice_feat)

        # Step 4: classify -> (B, num_classes)
        return self.classifier(patient_feat)


# ---------------------------------------------------------------------------
# segmentation head
# ---------------------------------------------------------------------------

class MultimodalMRISegHead(nn.Module):
    def __init__(self, sample_F_all, hidden_dim, patch_size=16, input_dim=4, output_dim=3):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modality_keys = list(sample_F_all.keys())
        self.num_blocks = len(next(iter(sample_F_all.values())))
        self.decoder = UNETRDecoder3D(
            embed_dim=hidden_dim, 
            patch_size=patch_size,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def forward(self, x, F_all):
        # F_all[k][i][0]: (B, D, num_patches, hidden_dim)
        encoder_features = [
            torch.cat([F_all[k][i][0] for k in self.modality_keys], dim=-1)
            for i in range(self.num_blocks)
        ]
        # encoder_features: list<len=num_blocks, shape=(B,D,num_patches,4*hidden_dim)>

        B, C, D, H, W = x.shape
        # output = self.decoder(x.view(B, D, C, H, W).contiguous(), encoder_features)
        output = self.decoder(x.permute(0, 2, 1, 3, 4).contiguous(), encoder_features)
        return output
