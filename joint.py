"""
RNN-T Joint Network
 
Combines encoder output and decoder output into per-token logits.
 
    joint(encoder_out, decoder_out) = Linear(Tanh(Linear(enc) + Linear(dec)))
 
Output shape: (B, T, U+1, vocab_size)
This is what the RNN-T loss consumes.
"""
 
import torch
import torch.nn as nn
 
 
class RNNTJointNetwork(nn.Module):
    """
    Args:
        encoder_dim  : d_model of the Conformer encoder
        decoder_dim  : hidden size of the LSTM decoder
        joint_dim    : inner projection dimension
        vocab_size   : number of output tokens (including blank=0)
    """
 
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joint_dim:   int,
        vocab_size:  int,
    ):
        super().__init__()
        self.enc_proj  = nn.Linear(encoder_dim, joint_dim)
        self.dec_proj  = nn.Linear(decoder_dim, joint_dim)
        self.output    = nn.Linear(joint_dim, vocab_size)
 
    def forward(
        self,
        encoder_out: torch.Tensor,   # (B, T, encoder_dim)
        decoder_out: torch.Tensor,   # (B, U+1, decoder_dim)
    ) -> torch.Tensor:
        """
        Returns:
            logits : (B, T, U+1, vocab_size)
        """
        # Expand dims to broadcast over T and U+1
        enc = self.enc_proj(encoder_out)          # (B, T,   joint_dim)
        dec = self.dec_proj(decoder_out)          # (B, U+1, joint_dim)
 
        enc = enc.unsqueeze(2)                    # (B, T, 1,   joint_dim)
        dec = dec.unsqueeze(1)                    # (B, 1, U+1, joint_dim)
 
        joint  = torch.tanh(enc + dec)            # (B, T, U+1, joint_dim)
        logits = self.output(joint)               # (B, T, U+1, vocab_size)
        return logits
 
 
# sanity check
if __name__ == "__main__":
    joint = RNNTJointNetwork(encoder_dim=144, decoder_dim=320,
                             joint_dim=320, vocab_size=30)
    enc = torch.randn(2, 10, 144)   # B=2, T=10 frames
    dec = torch.randn(2,  6, 320)   # B=2, U+1=6 tokens
    out = joint(enc, dec)
    print(f"Encoder : {enc.shape}")
    print(f"Decoder : {dec.shape}")
    print(f"Logits  : {out.shape}")  # (2, 10, 6, 30)
 