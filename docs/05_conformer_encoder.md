# Conformer Encoder (Full Pipeline)

**File:** `modules/conformer_encoder.py`

---

## What is it, in plain English?

The full encoder is the **complete journey** audio takes before being decoded into text. Think of it as an assembly line:

```
Raw Audio
   ↓ (mic)
Log-Mel Filterbanks       ← convert sound waves to frequency features
   ↓
SpecAugment               ← randomly hide parts (teaches robustness)
   ↓
Conv Subsampling           ← compress time by 4× (speed up processing)
   ↓
Block 1  ┐
Block 2  │
Block 3  │  ← 16 or 17 Conformer Blocks stacked
  ...    │    each one refines the representation
Block 16 ┘
   ↓
Encoded Representation    ← ready for the decoder to predict text
```

---

## Stage 1: Log-Mel Filterbanks

**What:** Convert raw audio waveform to 80 frequency channels.

**Why:** Neural networks can't easily process raw audio samples (44,100 per second). Filterbanks compress this into 80 frequency "buckets" per 10ms window — much more manageable.

**With our sample data:**
```
Raw audio: [1, 2, 3, 4, ...]  (thousands of samples per second)
   ↓ 25ms FFT window, 10ms stride
Filterbanks: shape (T, 80)    ← T frames, each with 80 frequency values
```

---

## Stage 2: SpecAugment (Training Only)

**What:** Randomly mask frequency bands and time regions.

**Why:** Forces the model to learn robust representations — it can't rely on any single frequency or time window always being available. Like studying with some notes covered up.

**Paper settings:**
- Frequency masking: mask up to 27 frequency channels
- Time masking: 10 random time blocks, each up to 5% of utterance length

**Example (T=100 frames, F=80 channels):**
```
Before SpecAugment:
████████████████████████████████████████  ← all frames visible
████████████████████████████████████████
...

After SpecAugment:
████████████████████████████████████████
████████████████████████████████████████
████░░░░░░░░░░████████████████████████░░  ← some channels masked (grey)
████░░░░░░░░░░████████████████████████░░
████████░░░░░░████████████░░░░░████████   ← some time steps masked
```

At inference time, SpecAugment is **switched off** — the model sees full audio.

---

## Stage 3: Convolution Subsampling

**What:** Two stacked 2D convolutions with stride 2 each, reducing time by 4×.

**Why:** 40ms is sufficient temporal resolution for speech recognition. Processing every 10ms frame through 16 attention layers would be very slow.

**Shape transformation:**
```
Input:  (B, T,    80)
  ↓ unsqueeze
        (B, 1, T, 80)   ← treat as single-channel image
  ↓ Conv2D stride=2
        (B, d, T/2, 39)
  ↓ Conv2D stride=2
        (B, d, T/4, 19)
  ↓ reshape + linear
Output: (B, T/4, d_model)
```

**With our demo (T=40 frames):**
```
Input:  (2, 40, 80)
Output: (2,  9, 144)    ← 40/4 ≈ 9 frames after subsampling
```

---

## Stage 4: N × Conformer Blocks

**What:** Stack 16 (or 17) Conformer blocks, each refining the representation.

**Input to block 1:** raw subsampled features
**Output of block 16:** rich, context-aware encodings ready for decoding

Each block's output becomes the next block's input — the representation gets progressively refined.

---

## Model Sizes (Table 1 from paper)

| Size | d_model | Heads | Layers | Params | WER (test-clean) |
|------|---------|-------|--------|--------|-----------------|
| **S** | 144 | 4 | 16 | 10.3M | 2.7% |
| **M** | 256 | 4 | 16 | 30.7M | 2.3% |
| **L** | 512 | 8 | 17 | 118.8M | 2.1% |

The M model (30M params) **already beats** the Transformer Transducer with 139M params.

---

## Full Walkthrough: Mini Batch

**Input:** 2 utterances, 40 frames each, 80 filterbank channels
```python
x = torch.randn(2, 40, 80)   # (batch=2, time=40, freq=80)
```

**After subsampling:**
```
(2, 40, 80) → (2, 9, 144)
```
Time compressed 4×, projected to d_model=144.

**After 16 Conformer blocks:**
```
(2, 9, 144) → (2, 9, 144)   ← shape unchanged, but representation deeply refined
```

**Output:** Each of the 9 time steps now contains a rich 144-dimensional vector encoding:
- What sound is happening now (local, from conv)
- What sounds came before and after (global, from attention)
- How this sound relates to the whole utterance

---

## Actual Output (from running the code)

```
Input  shape : (2, 40, 80)   ← 2 utterances, 40 frames, 80 filterbanks
Output shape : (2,  9, 144)  ← 2 utterances, ~9 frames, 144-dim encoding

First encoded vector (utterance 0, frame 0):
[0.165, -0.198, -0.648, -1.330, 0.272, 0.250, -0.397, 0.294, ...]
```

These 144 numbers represent everything the model has learned about that 40ms of audio in context.

---

## What Comes After the Encoder?

The encoder output feeds into a **decoder** — in the paper, a single-LSTM decoder that predicts text tokens (using RNN-T loss). The decoder asks:

> *"Given these encoded audio representations, what's the most likely word sequence?"*

That decoder is a separate component — the Conformer is purely the encoder.

---

## How to Build Each Model Size

```python
from conformer_encoder import build_conformer

model_s = build_conformer("S")   # 10M  params
model_m = build_conformer("M")   # 30M  params
model_l = build_conformer("L")   # 118M params
```
