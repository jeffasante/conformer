
"""
Toy dataset — 100 synthetic samples for training loop verification.
 
Each sample is:
  - A random filterbank tensor  (T, 80) — simulates real audio features
  - A random token sequence     (U,)    — simulates a transcript
 
In a real setup you would replace this with LibriSpeech or Twi data,
loading actual audio and running it through a filterbank extractor.
"""
 
import torch
from torch.utils.data import Dataset, DataLoader
 
 
CHARS = list("abcdefghijklmnopqrstuvwxyz '")   # 28 chars
# token index 0 = blank (reserved for RNN-T)
# token index 1..28 = characters
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i+1: c for i, c in enumerate(CHARS)}
VOCAB_SIZE  = len(CHARS) + 1   # +1 for blank at index 0
 
# sample sentences (stand-in for real transcripts)  
SENTENCES = [
    "hello world",
    "good morning",
    "speech recognition",
    "conformer model",
    "attention mechanism",
    "convolution layer",
    "feed forward network",
    "layer normalization",
    "batch normalization",
    "dropout regularization",
]

def text_to_tokens(text: str) -> list[int]:
    """Convert text string to list of token indices (no blank)."""
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]
 
 
def tokens_to_text(tokens: list[int]) -> str:
    """Convert token indices back to string."""
    return "".join(IDX_TO_CHAR.get(t, "?") for t in tokens if t != 0)
 
class ToyASRDataset(Dataset):
    """
    100 synthetic ASR samples.
 
    Each sample has:
      features       : (T, 80)  log-mel filterbank features (random, simulated)
      feature_length : int      number of valid frames
      tokens         : list[int] transcript as token indices
      token_length   : int      transcript length
 
    T varies between 60 and 160 frames (600ms to 1.6s of audio).
    """
 
    def __init__(self, n_samples: int = 100, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
 
        self.samples = []
        for i in range(n_samples):
            # Pick a random sentence and repeat/extend to fill 100 samples
            text = SENTENCES[i % len(SENTENCES)]
 
            # Random frame length between 60 and 160
            T = torch.randint(60, 161, (1,), generator=rng).item()
 
            # Simulate filterbank features: (T, 80)
            # Real features would come from torchaudio.transforms.MelSpectrogram
            features = torch.randn(T, 80, generator=rng) * 0.5
 
            tokens = text_to_tokens(text)
 
            self.samples.append({
                "features"      : features,
                "feature_length": T,
                "tokens"        : torch.tensor(tokens, dtype=torch.long),
                "token_length"  : len(tokens),
                "text"          : text,
            })
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        return self.samples[idx]
 
 
def collate_fn(batch):
    """
    Pad variable-length features and tokens to the max length in the batch.
    """
    # Sort by feature length descending (helps RNN-T loss)
    batch = sorted(batch, key=lambda x: x["feature_length"], reverse=True)
 
    max_T = max(s["feature_length"] for s in batch)
    max_U = max(s["token_length"]   for s in batch)
    B     = len(batch)
 
    features        = torch.zeros(B, max_T, 80)
    feature_lengths = torch.zeros(B, dtype=torch.long)
    targets         = torch.zeros(B, max_U, dtype=torch.long)
    target_lengths  = torch.zeros(B, dtype=torch.long)
 
    for i, s in enumerate(batch):
        T = s["feature_length"]
        U = s["token_length"]
        features[i, :T, :]  = s["features"]
        feature_lengths[i]  = T
        targets[i, :U]      = s["tokens"]
        target_lengths[i]   = U
 
    return features, feature_lengths, targets, target_lengths
 
 
def get_dataloader(n_samples=100, batch_size=8, shuffle=True):
    dataset = ToyASRDataset(n_samples=n_samples)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_fn,
    )
 
 
# sanity check
if __name__ == "__main__":
    ds = ToyASRDataset(n_samples=100)
    print(f"Dataset size : {len(ds)}")
    print(f"Vocab size   : {VOCAB_SIZE}")
    print(f"Sample 0     : '{ds[0]['text']}'")
    print(f"  tokens     : {ds[0]['tokens'].tolist()}")
    print(f"  features   : {ds[0]['features'].shape}")
 
    loader = get_dataloader(batch_size=4)
    features, feat_len, targets, tgt_len = next(iter(loader))
    print(f"\nBatch features : {features.shape}")
    print(f"Feat lengths   : {feat_len.tolist()}")
    print(f"Targets        : {targets.shape}")
    print(f"Target lengths : {tgt_len.tolist()}")