"""
Conformer Encoder — from the Conformer paper (Gulati et al., 2020)

Full pipeline (Figure 1 in paper):
    Audio filterbanks (B, T, 80)
    → SpecAugment (training only)
    → Conv Subsampling  (B, T/4, d_model)
    → Linear projection + Dropout
    → N × ConformerBlock
    → Output (B, T/4, d_model)

Three model sizes from Table 1:
    Conformer-S : d=144, heads=4, layers=16,  ~10M  params
    Conformer-M : d=256, heads=4, layers=16,  ~30M  params
    Conformer-L : d=512, heads=8, layers=17,  ~118M params
"""

import torch
import torch.nn as nn

from conformer_block import ConformerBlock


# Convolution subsampling (reduces time by factor 4)
class ConvSubsampling(nn.Module):
    """
    Two stacked Conv2D layers that reduce the time resolution by 4×.
    Input is treated as a single-channel "image": (B, 1, T, 80).
    Output is projected to d_model.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),   # halve T and F
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),  # halve again
            nn.ReLU(),
        )
        # After two stride-2 convs on freq dim 80:
        #   floor((floor((80-3)/2+1)-3)/2+1) = 19
        self.proj = nn.Linear(d_model * 19, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 80)  — log-mel filterbank features
        Returns:
            out : (B, T', d_model)  where T' ≈ T/4
        """
        B, T, F = x.shape
        x = x.unsqueeze(1)               # (B, 1, T, F)
        x = self.conv(x)                 # (B, d_model, T', F')
        B2, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3)       # (B, T', d_model, F')
        x = x.contiguous().view(B2, T2, C * F2)   # (B, T', d_model*F')
        x = self.proj(x)                 # (B, T', d_model)
        return x


# SpecAugment (training-only data augmentation)
class SpecAugment(nn.Module):
    """
    Masks random frequency bands and time steps (Park et al., 2019).
    Paper settings: F=27 freq mask, 10 time masks with pS=0.05.
    Applied to raw filterbank features before subsampling.
    """

    def __init__(self, freq_mask_param: int = 27, n_time_masks: int = 10, time_mask_ratio: float = 0.05):
        super().__init__()
        self.F  = freq_mask_param
        self.nT = n_time_masks
        self.pS = time_mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, F)"""
        if not self.training:
            return x

        B, T, F = x.shape
        x = x.clone()

        # Frequency masking
        f  = torch.randint(0, self.F, (1,)).item()
        f0 = torch.randint(0, F - f, (1,)).item()
        x[:, :, f0: f0 + f] = 0.0

        # Time masking
        max_t = int(self.pS * T)
        for _ in range(self.nT):
            t  = torch.randint(0, max(1, max_t), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x[:, t0: t0 + t, :] = 0.0

        return x


# Full Conformer Encoder
class ConformerEncoder(nn.Module):
    """
    Full Conformer encoder.

    Args:
        d_model      : model dimension
        n_heads      : attention heads
        n_layers     : number of Conformer blocks (16 or 17)
        ffn_expansion: FFN inner expansion (4)
        kernel_size  : depthwise conv kernel (32)
        dropout      : dropout rate (0.1)
        input_dim    : number of filterbank channels (80)
    """

    def __init__(
        self,
        d_model:      int   = 256,
        n_heads:      int   = 4,
        n_layers:     int   = 16,
        ffn_expansion: int   = 4,
        kernel_size:  int   = 32,
        dropout:      float = 0.1,
        input_dim:    int   = 80,
    ):
        super().__init__()

        self.spec_aug   = SpecAugment()
        self.subsampling = ConvSubsampling(d_model)
        self.dropout    = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ffn_expansion, kernel_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : (batch, time, 80)  — log-mel filterbanks
            mask : optional padding mask
        Returns:
            out  : (batch, time/4, d_model)
        """
        x = self.spec_aug(x)         # augment (no-op at eval)
        x = self.subsampling(x)      # (B, T/4, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        return x


# Model size configs (Table 1 from paper)
CONFIGS = {
    "S": dict(d_model=144, n_heads=4, n_layers=16, kernel_size=32),
    "M": dict(d_model=256, n_heads=4, n_layers=16, kernel_size=32),
    "L": dict(d_model=512, n_heads=8, n_layers=17, kernel_size=32),
}

def build_conformer(size: str = "S", **kwargs) -> ConformerEncoder:
    cfg = {**CONFIGS[size], **kwargs}
    return ConformerEncoder(**cfg)


# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate a mini batch: 2 utterances, 40 frames each, 80 filterbank channels
    # (Real audio would have thousands of frames)
    x = torch.randn(2, 40, 80)

    print("=== Full Conformer Encoder (Conformer-S, tiny demo) ===\n")
    model = ConformerEncoder(
        d_model=144,
        n_heads=4,
        n_layers=2,       # use 2 instead of 16 for quick test
        kernel_size=3,    # use 3 instead of 32 for tiny T
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        out = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input  shape : {x.shape}   (batch=2, T=40, filterbanks=80)")
    print(f"Output shape : {out.shape}  (batch=2, T'≈T/4, d_model=144)")
    print(f"Parameters   : {total_params:,}")
    print(f"\nFirst output vector (batch 0, frame 0):")
    print([round(v, 4) for v in out[0, 0].tolist()[:8]], "...")
