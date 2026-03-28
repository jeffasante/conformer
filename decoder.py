"""
RNN-T Decoder (Prediction Network) — paper Section 3.2
 
The paper uses a single-LSTM-layer decoder.
It takes the previously predicted token and outputs a hidden state
that is combined with the encoder output in the Joint Network.
 
Architecture:
    Embedding(vocab_size, d_model)
    → LSTM(d_model, decoder_dim, num_layers=1)
    → output: (batch, 1, decoder_dim)
"""
 
import torch
import torch.nn as nn


class RNNTDecoder(nn.Module):
    """
    Single-layer LSTM prediction network for RNN-T.
 
    Args:
        vocab_size   : number of tokens (characters + blank)
        embed_dim    : embedding dimension (= d_model for simplicity)
        decoder_dim  : LSTM hidden size (320 for S, 640 for M/L — Table 1)
        num_layers   : paper uses 1
    """
 
    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int  = 144,
        decoder_dim:  int  = 320,
        num_layers:   int  = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = decoder_dim,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.decoder_dim = decoder_dim
 
    def forward(
        self,
        targets:    torch.Tensor,                        # (B, U)
        hidden:     tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            targets : (B, U)  — token indices (SOS-prepended during training)
            hidden  : optional LSTM hidden state for streaming
        Returns:
            out     : (B, U, decoder_dim)
            hidden  : updated LSTM state
        """
        emb = self.embedding(targets)           # (B, U, embed_dim)
        out, hidden = self.lstm(emb, hidden)    # (B, U, decoder_dim)
        return out, hidden

# sanity check
if __name__ == "__main__":
    vocab   = 30
    decoder = RNNTDecoder(vocab_size=vocab, embed_dim=144, decoder_dim=320)
    targets = torch.randint(0, vocab, (2, 5))   # batch=2, U=5 tokens
    out, _  = decoder(targets)
    print(f"Input  : {targets.shape}")
    print(f"Output : {out.shape}")   # (2, 5, 320)
 