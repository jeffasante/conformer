"""
rel_attention.py
Exact implementation of Relative Multi-Head Self-Attention (RelMSHA)
from the Conformer paper (Gulati et al., 2020) and Transformer-XL.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelMultiHeadSelfAttention(nn.Module):
    """
    Relative Multi-Head Self-Attention as used in Google Conformer.
    This implements the 4-term dot product to achieve shift-invariance.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Key, Query, Value projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        # (Piece #1) Linear projection for positional encodings
        self.linear_pos = nn.Linear(d_model, d_model)

        # (Piece #2) Learned bias vectors (The 'u' and 'v' from the paper)
        # These represent global content bias and global position bias.
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_heads, self.d_k))
        
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)

    def rel_shift(self, x: torch.Tensor):
        """
        The 'Relative Shift' trick for bidirectional (non-causal) attention.
        Input x: (B, H, T, 2T-1)
        Goal: Extract (B, H, T, T) such that element (i, j) comes from 
              relative distance (i-j).
        """
        B, H, T, L = x.size()
        
        # Pad to (B, H, T, 2T) to make it easy to reshape
        x = F.pad(x, (0, 1))
        # Reshape and shift to align relative positions
        x = x.view(B, H, 2 * T, T)
        x = x[:, :, T:, :] # Take the last T rows
        return x.view(B, H, T, T)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x       : Content input (B, T, d_model)
            pos_emb : Sinusoidal Positional Encodings (1, 2T-1, d_model)
            mask    : Attention mask (B, 1, T, T)
        """
        B, T, _ = x.size()
        
        # Project Q, K, V
        # (B, T, d_model) -> (B, H, T, d_k)
        q = self.linear_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Project Positional Encoding
        # (1, 2T-1, d_model) -> (1, 2T-1, H, d_k)
        p = self.linear_pos(pos_emb).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Calculate the 4 Attention Terms
        # Term (a) + (c): Content-Content + Global Content Bias
        # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        content_score = torch.matmul(q + self.pos_bias_u.unsqueeze(1), k.transpose(-2, -1))

        # Term (b) + (d): Content-Position + Global Position Bias
        # (B, H, T, d_k) * (1, H, d_k, 2T-1) -> (B, H, T, 2T-1)
        pos_score = torch.matmul(q + self.pos_bias_v.unsqueeze(1), p.transpose(-2, -1))
        
        # Apply 'RelShift' to term (b+d) to align with (a+c)
        pos_score = self.rel_shift(pos_score)

        # Combined Score
        scores = (content_score + pos_score) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6. Final Output
        # (B, H, T, T) * (B, H, T, d_k) -> (B, H, T, d_k)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.linear_out(out)

# SINUSOIDAL POSITIVE GENERATOR
def get_sinusoidal_embeddings(length: int, d_model: int):
    """Generates relative sinusoidal encodings for length 2T-1"""
    half = d_model // 2
    pe = torch.zeros(length, d_model)
    pos = torch.arange(-(length // 2), length // 2 + 1).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0) # (1, L, d_model)


if __name__ == "__main__":
    B, T, D, H = 1, 50, 256, 4
    x = torch.randn(B, T, D)
    pos_emb = get_sinusoidal_embeddings(2 * T - 1, D)
    
    model = RelMultiHeadSelfAttention(d_model=D, n_heads=H)
    output = model(x, pos_emb)
    
    print(f"Input shape     : {x.shape}")
    print(f"Pos Emb shape   : {pos_emb.shape}")
    print(f"Output shape    : {output.shape}")
    print("\nThis version uses learned 'u' and 'v' parameters and the 'RelShift' logic.")
