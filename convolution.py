"""
Convolution Module — from the Conformer paper (Gulati et al., 2020)

Architecture (Figure 2 in paper):
    LayerNorm
    → Pointwise Conv (d → 2d)          [expansion factor 2]
    → GLU                               [gates half the channels → back to d]
    → 1-D Depthwise Conv (kernel=32)    [captures local patterns]
    → BatchNorm
    → Swish
    → Pointwise Conv (d → d)
    → Dropout

The residual is added by the caller: x'' = x' + Conv(x')
"""

import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module.

    Args:
        d_model     : model dimension
        kernel_size : depthwise conv kernel size (paper uses 32)
        dropout     : dropout rate (0.1)
    """

    def __init__(self, d_model: int, kernel_size: int = 32, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd for same-length output"

        self.norm = nn.LayerNorm(d_model)

        # Pointwise conv 1: expand d → 2d (for GLU)
        self.pw_conv1 = nn.Conv1d(
            in_channels  = d_model,
            out_channels = 2 * d_model,
            kernel_size  = 1,
        )

        # GLU: splits 2d into two halves; no learned params
        self.glu = nn.GLU(dim=1)   # splits channel dim → output is d

        # 1-D Depthwise conv: each channel convolved independently
        self.dw_conv = nn.Conv1d(
            in_channels  = d_model,
            out_channels = d_model,
            kernel_size  = kernel_size,
            padding      = (kernel_size - 1) // 2,   # "same" padding
            groups       = d_model,                   # depthwise = groups==channels
        )

        self.bn    = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()   # SiLU == Swish

        # Pointwise conv 2: project back d → d
        self.pw_conv2 = nn.Conv1d(
            in_channels  = d_model,
            out_channels = d_model,
            kernel_size  = 1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, time, d_model)
        Returns:
            out : (batch, time, d_model)  — no residual; caller adds it
        """
        # Pre-norm
        out = self.norm(x)                         # (B, T, d)

        # Conv1d expects (B, C, T) — so transpose
        out = out.transpose(1, 2)                  # (B, d, T)

        # Pointwise expand
        out = self.pw_conv1(out)                   # (B, 2d, T)

        # GLU gate: A * sigmoid(B) where A,B are the two halves
        out = self.glu(out)                        # (B, d, T)

        # Depthwise conv — captures LOCAL patterns in time
        out = self.dw_conv(out)                    # (B, d, T)

        # BatchNorm + Swish
        out = self.bn(out)
        out = self.swish(out)

        # Pointwise project back
        out = self.pw_conv2(out)                   # (B, d, T)

        out = self.dropout(out)

        # Transpose back to (B, T, d)
        out = out.transpose(1, 2)                  # (B, T, d)
        return out   # caller does:  x = x + conv(x)


# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, d = 1, 4, 8

    # 4 audio frames, each represented as an 8-dim vector
    x = torch.tensor([[
        [1., 2., 3., 4., 1., 2., 3., 4.],   # frame 0
        [0., 1., 0., 1., 0., 1., 0., 1.],   # frame 1
        [2., 2., 2., 2., 2., 2., 2., 2.],   # frame 2
        [4., 3., 2., 1., 4., 3., 2., 1.],   # frame 3
    ]])  # (1, 4, 8)

    # Use kernel_size=3 for tiny demo (paper uses 32 on real data)
    conv = ConvolutionModule(d_model=8, kernel_size=3, dropout=0.0)
    conv.eval()

    # Show intermediate shapes step by step
    normed = conv.norm(x)
    transposed = normed.transpose(1, 2)                  # (B, d, T)
    after_pw1  = conv.pw_conv1(transposed)               # (B, 2d, T)
    after_glu  = conv.glu(after_pw1)                     # (B, d, T)
    after_dw   = conv.dw_conv(after_glu)                 # (B, d, T)
    after_bn   = conv.bn(after_dw)
    after_sw   = conv.swish(after_bn)
    after_pw2  = conv.pw_conv2(after_sw)                 # (B, d, T)
    final      = after_pw2.transpose(1, 2)               # (B, T, d)
    with_res   = x + final                               # residual

    print("=== Convolution Module — step-by-step shapes ===")
    print(f"Input              : {x.shape}")
    print(f"After LayerNorm    : {normed.shape}")
    print(f"After transpose    : {transposed.shape}  ← (B, d, T) for Conv1d")
    print(f"After PW-Conv1 (×2): {after_pw1.shape}  ← expand to 2d")
    print(f"After GLU          : {after_glu.shape}   ← gate back to d")
    print(f"After DW-Conv      : {after_dw.shape}   ← local pattern capture")
    print(f"After BN + Swish   : {after_sw.shape}")
    print(f"After PW-Conv2     : {after_pw2.shape}")
    print(f"After transpose    : {final.shape}   ← back to (B, T, d)")
    print(f"With residual      : {with_res.shape}")

    print("\nFrame 0 input  :", x[0, 0].tolist())
    print("Frame 0 conv   :", [round(v, 4) for v in final[0, 0].tolist()])
    print("Frame 0 output :", [round(v, 4) for v in with_res[0, 0].tolist()])
