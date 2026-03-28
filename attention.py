"""
Multi-Head Self-Attention Module (MHSA) — from the Conformer paper (Gulati et al., 2020)

Architecture (Figure 3 in paper):
    LayerNorm → Multi-Head Attention (with Relative Positional Encoding) → Dropout

Key design choices:
  - Pre-norm (LayerNorm applied to input BEFORE attention, not after)
  - Relative sinusoidal positional encoding (from Transformer-XL, Dai et al. 2019)
    This helps the model generalise to variable-length audio inputs
  - Residual added by the caller: x' = x + MHSA(x)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal relative positional encoding
def get_relative_position_encoding(length: int, d_model: int) -> torch.Tensor:
    """
    Build sinusoidal encodings for relative positions 0 … length-1.

    Returns:
        pe : (length, d_model)  — one row per position
    """
    pe  = torch.zeros(length, d_model)
    pos = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)   # (L, 1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)   # even dims  → sin
    pe[:, 1::2] = torch.cos(pos * div)   # odd  dims  → cos
    return pe   # (length, d_model)


# MHSA Module
class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-headed self-attention with relative positional encoding and pre-norm.

    Args:
        d_model  : model dimension (e.g. 144 / 256 / 512)
        n_heads  : number of attention heads (4 / 4 / 8 for S/M/L)
        dropout  : dropout rate (0.1 in paper)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # dimension per head

        self.norm     = nn.LayerNorm(d_model)
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, n_heads, T, d_k)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # (B, H, T, d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : (batch, time, d_model)
            mask : optional boolean mask (batch, 1, 1, time) — True = ignore
        Returns:
            out  : (batch, time, d_model) — no residual; caller adds it
        """
        B, T, _ = x.shape

        # pre-norm
        normed = self.norm(x)

        # project to Q, K, V and split heads
        Q = self._split_heads(self.q_proj(normed))   # (B, H, T, d_k)
        K = self._split_heads(self.k_proj(normed))
        V = self._split_heads(self.v_proj(normed))

        # relative positional bias (simplified)
        # Full Transformer-XL RPE requires per-query relative biases;
        # here we add sinusoidal encoding directly to K for clarity.
        pe = get_relative_position_encoding(T, self.d_k).to(x.device)  # (T, d_k)
        K  = K + pe.unsqueeze(0).unsqueeze(0)   # broadcast over B, H

        # scaled dot-product attention
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B,H,T,T)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn   = F.softmax(scores, dim=-1)   # (B, H, T, T)
        attn   = self.dropout(attn)

        # weighted sum over V
        context = torch.matmul(attn, V)             # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous()
        context = context.view(B, T, self.d_model)  # (B, T, d_model)

        out = self.out_proj(context)
        out = self.dropout(out)
        return out   # caller does:  x = x + mhsa(x)


# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, d, H = 1, 4, 8, 2   # 1 batch, 4 time-steps, dim=8, 2 heads

    # Each row = one time-step (like one 10ms audio frame)
    x = torch.tensor([[
        [1., 2., 3., 4., 1., 2., 3., 4.],   # frame 0
        [0., 1., 0., 1., 0., 1., 0., 1.],   # frame 1
        [2., 2., 2., 2., 2., 2., 2., 2.],   # frame 2
        [4., 3., 2., 1., 4., 3., 2., 1.],   # frame 3
    ]])  # (1, 4, 8)

    mhsa = MultiHeadSelfAttentionModule(d_model=8, n_heads=2, dropout=0.0)
    mhsa.eval()

    raw = mhsa(x)
    out = x + raw   # full-step residual (Conformer block also adds this)

    print("Input  shape:", x.shape)
    print("MHSA   shape:", raw.shape)
    print("Output shape:", out.shape)
    print("\nFrame 0 input :", x[0, 0].tolist())
    print("Frame 0 MHSA  :", [round(v, 4) for v in raw[0, 0].tolist()])
    print("Frame 0 output:", [round(v, 4) for v in out[0, 0].tolist()])
