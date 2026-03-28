"""
Feed Forward Module (FFN) — from the Conformer paper (Gulati et al., 2020)

Architecture (Figure 4 in paper):
    LayerNorm → Linear(d → 4d) → Swish → Dropout → Linear(4d → d) → Dropout

Used TWICE per Conformer block in a "Macaron" style:
    - Once BEFORE the attention/conv modules (half-step residual)
    - Once AFTER  the attention/conv modules (half-step residual)

Half-step residual means: output = x + 0.5 * FFN(x)
"""

import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm, Swish activation, and dropout.

    Args:
        d_model   : model dimension (e.g. 144 for Conformer-S)
        expansion : inner expansion factor — paper uses 4
        dropout   : dropout rate — paper uses 0.1
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()

        self.norm   = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)   # d → 4d
        self.swish   = nn.SiLU()                                  # SiLU == Swish
        self.drop1   = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion, d_model)   # 4d → d
        self.drop2   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, time, d_model)
        Returns:
            out : (batch, time, d_model)  — same shape, no residual here
                  The caller adds the half-step residual: x + 0.5 * ffn(x)
        """
        out = self.norm(x)          # pre-norm
        out = self.linear1(out)     # d → 4d
        out = self.swish(out)       # Swish activation
        out = self.drop1(out)
        out = self.linear2(out)     # 4d → d
        out = self.drop2(out)
        return out


# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    batch, time, d = 2, 4, 8   # tiny dims for readability
    x = torch.tensor([
        [[1., 2., 3., 4., 1., 2., 3., 4.],
         [0., 1., 0., 1., 0., 1., 0., 1.],
         [2., 2., 2., 2., 2., 2., 2., 2.],
         [4., 3., 2., 1., 4., 3., 2., 1.]],
        [[1., 1., 1., 1., 1., 1., 1., 1.],
         [2., 2., 2., 2., 2., 2., 2., 2.],
         [3., 3., 3., 3., 3., 3., 3., 3.],
         [4., 4., 4., 4., 4., 4., 4., 4.]],
    ])  # shape (2, 4, 8)

    ffn = FeedForwardModule(d_model=8, expansion=4, dropout=0.0)
    ffn.eval()

    raw  = ffn(x)
    half = x + 0.5 * raw          # half-step residual (as used in Conformer block)

    print("Input shape        :", x.shape)
    print("FFN output shape   :", raw.shape)
    print("After half-residual:", half.shape)
    print("\nFirst token (batch 0, time 0):")
    print("  x      =", x[0, 0].tolist())
    print("  ffn(x) =", [round(v, 4) for v in raw[0, 0].tolist()])
    print("  result =", [round(v, 4) for v in half[0, 0].tolist()])
