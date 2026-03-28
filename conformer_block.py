"""
Conformer Block — from the Conformer paper (Gulati et al., 2020)

This is the core repeating unit. It stacks all 4 sub-modules in the
"Macaron" pattern described in Section 2.4 and Equation (1):

    x̃  = x  + 0.5 * FFN(x)          ← first  feed-forward (half-step)
    x' = x̃  + MHSA(x̃)               ← self-attention
    x''= x' + Conv(x')               ← convolution
    y  = LayerNorm(x'' + 0.5*FFN(x''))  ← second feed-forward (half-step) + norm
"""

import torch
import torch.nn as nn

from feed_forward import FeedForwardModule
from attention    import MultiHeadSelfAttentionModule
from convolution  import ConvolutionModule
from rel_attention import RelMultiHeadSelfAttention, get_sinusoidal_embeddings


class ConformerBlock(nn.Module):
    """
    Single Conformer encoder block.

    Args:
        d_model     : model dimension (144 / 256 / 512 for S/M/L)
        n_heads     : attention heads (4 / 4 / 8)
        ffn_expansion: FFN inner expansion factor (paper = 4)
        kernel_size : depthwise conv kernel size (paper = 32)
        dropout     : dropout rate (paper = 0.1)
    """

    def __init__(
        self,
        d_model:      int,
        n_heads:      int,
        ffn_expansion: int   = 4,
        kernel_size:  int   = 32,
        dropout:      float = 0.1,
        rel_attention: bool  = False,
    ):
        super().__init__()
        self.rel_attention = rel_attention
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        if rel_attention:
            self.mhsa = RelMultiHeadSelfAttention(d_model, n_heads, dropout)
        else:
            self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements Equation (1) from the paper.

        Args:
            x    : (batch, time, d_model)
            mask : optional attention mask
        Returns:
            y    : (batch, time, d_model)
        """
        # Step 1 — first Feed-Forward (half-step residual)
        x = x + 0.5 * self.ffn1(x) #  x^i + 1/2 FFN(x^i)

        # Step 2 — Multi-Head Self-Attention (full residual)
        if self.rel_attention:
            pos_emb = get_sinusoidal_embeddings(2 * x.size(1) - 1, x.size(-1)).to(x.device)
            x = x + self.mhsa(x, pos_emb, mask)
        else:
            x = x + self.mhsa(x, mask)

        # Step 3 — Convolution module (full residual)
        x = x + self.conv(x)

        # Step 4 — second Feed-Forward (half-step residual) + LayerNorm
        x = self.norm(x + 0.5 * self.ffn2(x))

        return x


# Quick sanity check
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, d = 1, 4, 8

    x = torch.tensor([[
        [1., 2., 3., 4., 1., 2., 3., 4.],   # frame 0
        [0., 1., 0., 1., 0., 1., 0., 1.],   # frame 1
        [2., 2., 2., 2., 2., 2., 2., 2.],   # frame 2
        [4., 3., 2., 1., 4., 3., 2., 1.],   # frame 3
    ]])  # (1, 4, 8)

    block = ConformerBlock(d_model=8, n_heads=2, kernel_size=3, dropout=0.0)
    block.eval()

    # Trace each step manually
    after_ffn1 = x + 0.5 * block.ffn1(x)
    after_mhsa = after_ffn1 + block.mhsa(after_ffn1)
    after_conv = after_mhsa + block.conv(after_mhsa)
    after_ffn2 = block.norm(after_conv + 0.5 * block.ffn2(after_conv))

    print("=== Conformer Block — step-by-step (Equation 1 from paper) ===\n")
    print(f"Input           : {x.shape}")
    print(f"After FFN-1     : {after_ffn1.shape}")
    print(f"After MHSA      : {after_mhsa.shape}")
    print(f"After Conv      : {after_conv.shape}")
    print(f"After FFN-2+LN  : {after_ffn2.shape}")

    print("\n--- Frame 0 values at each step ---")
    print("Input     :", x[0, 0].tolist())
    print("→ FFN-1   :", [round(v, 4) for v in after_ffn1[0, 0].tolist()])
    print("→ MHSA    :", [round(v, 4) for v in after_mhsa[0, 0].tolist()])
    print("→ Conv    :", [round(v, 4) for v in after_conv[0, 0].tolist()])
    print("→ FFN-2+LN:", [round(v, 4) for v in after_ffn2[0, 0].tolist()])
