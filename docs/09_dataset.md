# Toy ASR Dataset

**File:** `modules/dataset.py`

---

## What is it, in plain English?

The dataset is what feeds data into the training loop.

A real ASR dataset like LibriSpeech contains thousands of hours of audio paired with transcripts. For verifying that a training loop works, all of that is overkill. What you need is something that:

- Has variable-length inputs (like real audio)
- Has variable-length outputs (like real transcripts)
- Is fast to generate
- Is small enough to overfit on purpose (so you can confirm the model can learn)

This module generates 100 synthetic samples that satisfy all four.

---

## What Each Sample Contains

| Field | Type | Example |
|---|---|---|
| features | Tensor (T, 80) | random values simulating filterbanks |
| feature_length | int | 60 to 160 frames |
| tokens | Tensor (U,) | [8, 5, 12, 12, 15] for "hello" |
| token_length | int | 5 |
| text | string | "hello world" |

---

## The Vocabulary

```
blank (index 0)      ← reserved for RNN-T blank token
a     (index 1)
b     (index 2)
c     (index 3)
...
z     (index 26)
      (index 27)      ← space
'     (index 28)      ← apostrophe
```

Total vocab size: 29 tokens.

---

## Text to Tokens — Step by Step

Sample text: "hello"

### Step 1 — character mapping

```
h → index 8
e → index 5
l → index 12
l → index 12
o → index 15
```

### Step 2 — result

```
tokens = [8, 5, 12, 12, 15]
length = 5
```

This is what gets passed to the decoder during training.

---

## Simulated Filterbanks with [1, 2, 3, 4]

Real filterbanks come from applying a mel filter bank to a short-time Fourier transform of the audio waveform. For the toy dataset we skip that and just use random numbers.

Think of it this way: a real filterbank for one 10ms frame might look like:

```
frame 0: [1.2, 0.8, 2.1, 0.3, ..., 1.7]   ← 80 frequency values
frame 1: [0.9, 1.4, 1.1, 2.0, ..., 0.5]
frame 2: [1.8, 0.6, 0.4, 1.9, ..., 1.2]
frame 3: [0.4, 1.1, 2.3, 0.8, ..., 0.9]
```

Our toy dataset uses:
```python
features = torch.randn(T, 80) * 0.5
```

Random Gaussian values with standard deviation 0.5. Not real audio, but the same shape — which is all the model cares about for verifying the training pipeline.

---

## Padding and the Collate Function

Individual samples have different lengths:

```
Sample 0: features (120, 80), tokens length 11
Sample 1: features ( 94, 80), tokens length 18
Sample 2: features (155, 80), tokens length 10
Sample 3: features ( 67, 80), tokens length 13
```

But a training batch needs all samples to be the same shape (so they can be stacked into one tensor). The collate function pads shorter sequences with zeros:

### Before padding (batch of 4):

```
Sample 0: T=155, U=18
Sample 1: T=120, U=13
Sample 2: T= 94, U=11
Sample 3: T= 67, U=10
```

### After padding:

```
features shape: (4, 155, 80)   ← padded to max T=155
targets  shape: (4,  18)       ← padded to max U=18
```

The model only processes up to `feature_lengths[i]` frames for each sample in the batch. The zeros beyond that are ignored by the loss.

---

## Actual Output (from running the code)

```
Dataset size : 100 samples
Vocab size   : 29

Sample 0     : 'hello world'
  tokens     : [8, 5, 12, 12, 15, 27, 23, 15, 18, 12, 4]
  features   : torch.Size([94, 80])

Batch features : torch.Size([4, 154, 80])
Feat lengths   : [154, 140, 121, 90]
Targets        : torch.Size([4, 22])
Target lengths : [22, 18, 11, 10]
```

---

## Replacing This With Real Data

To use real audio instead of synthetic data, replace the features generation with:

```python
import torchaudio

waveform, sr = torchaudio.load("audio.wav")
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate    = sr,
    n_fft          = 400,    # 25ms window at 16kHz
    hop_length     = 160,    # 10ms stride
    n_mels         = 80,
)
features = mel(waveform).squeeze(0).T   # (T, 80)
```

Everything else in the pipeline stays the same.
