"""
Full Conformer Transducer model

Combines:
  - Conformer encoder  (this paper)
  - LSTM decoder       (single layer, paper Section 3.2)
  - Joint network      (standard RNN-T joint)
  - RNN-T loss         (torchaudio)
"""

import torch
import torch.nn as nn
from torchaudio.transforms import RNNTLoss

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conformer_encoder import ConformerEncoder
from decoder import RNNTDecoder
from joint import RNNTJointNetwork


class ConformerTransducer(nn.Module):
    """
    Conformer-S Transducer — matches paper Table 1 (small config).

    Args:
        vocab_size  : number of tokens including blank (index 0)
        d_model     : encoder dimension   (144 for S)
        n_heads     : attention heads     (4   for S)
        n_layers    : encoder layers      (16  for S)
        decoder_dim : LSTM hidden size    (320 for S)
        joint_dim   : joint network dim   (320 for S)
        kernel_size : depthwise conv k    (32  for S)
        dropout     : 0.1 throughout
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 144,
        n_heads:     int   = 4,
        n_layers:    int   = 16,
        decoder_dim: int   = 320,
        joint_dim:   int   = 320,
        kernel_size: int   = 32,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.encoder = ConformerEncoder(
            d_model     = d_model,
            n_heads     = n_heads,
            n_layers    = n_layers,
            kernel_size = kernel_size,
            dropout     = dropout,
        )
        self.decoder = RNNTDecoder(
            vocab_size  = vocab_size,
            embed_dim   = d_model,
            decoder_dim = decoder_dim,
        )
        self.joint = RNNTJointNetwork(
            encoder_dim = d_model,
            decoder_dim = decoder_dim,
            joint_dim   = joint_dim,
            vocab_size  = vocab_size,
        )
        self.loss_fn    = RNNTLoss(blank=0, clamp=-1, reduction="mean")
        self.vocab_size = vocab_size

    def forward(
        self,
        features:        torch.Tensor,   # (B, T, 80)
        feature_lengths: torch.Tensor,   # (B,)
        targets:         torch.Tensor,   # (B, U)   — token indices, no blank
        target_lengths:  torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Training forward pass. Returns scalar RNN-T loss.
        """
        # Encode audio
        encoder_out = self.encoder(features)              # (B, T', d_model)
        enc_lengths = self._encoder_lengths(feature_lengths)

        # Prepend SOS (blank=0) to targets for decoder input
        B   = targets.size(0)
        sos = torch.zeros(B, 1, dtype=torch.long, device=targets.device)
        dec_input = torch.cat([sos, targets], dim=1)      # (B, U+1)

        # Decode
        decoder_out, _ = self.decoder(dec_input)          # (B, U+1, decoder_dim)

        # Joint
        logits = self.joint(encoder_out, decoder_out)     # (B, T', U+1, vocab)

        # RNN-T loss  — expects log-softmax input
        log_probs = logits.log_softmax(dim=-1)

        loss = self.loss_fn(
            log_probs,
            targets.int(),
            enc_lengths.int(),
            target_lengths.int(),
        )
        return loss

    def _encoder_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map input frame count to encoder output frame count after 4x subsampling."""
        # Two stride-2 conv2d layers each reduce by roughly /2
        # Formula mirrors ConvSubsampling in conformer_encoder.py
        lengths = input_lengths
        for _ in range(2):
            lengths = (lengths - 3) // 2 + 1
        return lengths.clamp(min=1)

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,   # (1, T, 80)
        max_decode_len:  int = 200,
    ) -> list[int]:
        """
        Simple greedy decoding for inference.
        Returns list of token indices (no blank).
        """
        self.eval()
        encoder_out = self.encoder(features)   # (1, T', d)
        T = encoder_out.size(1)

        token = torch.zeros(1, 1, dtype=torch.long, device=features.device)
        hidden = None
        decoded = []

        for t in range(T):
            enc_t = encoder_out[:, t:t+1, :]           # (1, 1, d)
            dec_out, hidden = self.decoder(token, hidden)  # (1, 1, dec_dim)
            logits = self.joint(enc_t, dec_out)        # (1, 1, 1, vocab)
            pred   = logits.squeeze().argmax(-1).item()

            if pred != 0:   # 0 = blank
                decoded.append(pred)
                token = torch.tensor([[pred]], device=features.device)

            if len(decoded) >= max_decode_len:
                break

        return decoded


# sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    VOCAB = 30

    model = ConformerTransducer(
        vocab_size  = VOCAB,
        d_model     = 144,
        n_heads     = 4,
        n_layers    = 2,      # 2 instead of 16 for speed
        kernel_size = 3,
        dropout     = 0.0,
    )

    B, T, F = 2, 40, 80
    features        = torch.randn(B, T, F)
    feature_lengths = torch.tensor([40, 35])
    targets         = torch.randint(1, VOCAB, (B, 5))
    target_lengths  = torch.tensor([5, 4])

    loss = model(features, feature_lengths, targets, target_lengths)
    print(f"Features  : {features.shape}")
    print(f"Targets   : {targets.shape}")
    print(f"RNN-T loss: {loss.item():.4f}")

    # Greedy decode one sample
    tokens = model.greedy_decode(features[:1], max_decode_len=20)
    print(f"Decoded tokens: {tokens}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params  : {total:,}")
