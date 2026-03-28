# Conformer Transducer (Full Model)

**File:** `modules/conformer_transducer.py`

---

## What is it, in plain English?

The Conformer Transducer is the **complete speech recognition system**.

It connects all the pieces:

```
Audio features
    ↓
Conformer Encoder    ← "What does the audio sound like?"
    ↓
Joint Network   ←   Decoder  ← "What have I said so far?"
    ↓
RNN-T Loss / Greedy Decode
    ↓
Text
```

The word "Transducer" comes from the RNN-T (Recurrent Neural Network Transducer) loss function. It is the loss the paper uses because it can handle variable-length audio aligning to variable-length transcripts without needing a fixed alignment between them.

---

## Why RNN-T Instead of CTC?

CTC (Connectionist Temporal Classification) is simpler but has one constraint: it assumes each output token corresponds to one or more input frames in a strictly left-to-right order.

RNN-T removes that constraint. The model can emit tokens at any frame, and can emit multiple tokens per frame. This makes it better for:
- Streaming recognition (tokens emitted in real time)
- Languages with complex phoneme-to-grapheme mappings
- Long utterances

The paper uses RNN-T with a single LSTM decoder.

---

## Full Architecture

```
(B, T, 80) filterbanks
    ↓  SpecAugment
    ↓  Conv Subsampling  →  (B, T/4, 144)
    ↓  16 × ConformerBlock
encoder_out : (B, T', 144)

targets [a, b, c, d]
    ↓  Prepend SOS: [blank, a, b, c, d]
    ↓  Embedding
    ↓  Single LSTM
decoder_out : (B, U+1, 320)

encoder_out + decoder_out
    ↓  Joint Network
logits : (B, T', U+1, vocab_size)
    ↓  log_softmax
    ↓  RNN-T Loss
scalar loss
```

---

## The RNN-T Loss — Plain English

Imagine you have 4 audio frames and the target transcript is "ab".

The model needs to emit "a" and "b" somewhere across those 4 frames, using blank tokens to fill the rest. Valid alignments include:

```
Frame:    1      2      3      4
Align 1:  a      b      -      -
Align 2:  a      -      b      -
Align 3:  a      -      -      b
Align 4:  -      a      b      -
Align 5:  -      a      -      b
Align 6:  -      -      a      b
```

Where "-" is blank. There are 6 valid ways to align "ab" across 4 frames.

The RNN-T loss says: sum the probabilities of ALL valid alignments, take the log, negate it. Minimising this forces the model to assign high probability to correct transcripts regardless of where the tokens land.

---

## Step-by-Step with [1, 2, 3, 4]

Use token indices [1, 2, 3, 4] as a target transcript (4 tokens).

### Input

```
features       = random (1, 40, 80)    ← 1 sample, 40 audio frames
feature_length = [40]
targets        = [1, 2, 3, 4]          ← transcript
target_length  = [4]
```

### Step 1 — Encoder

```
(1, 40, 80)
  → subsampling
(1,  9, 144)    ← ~40/4 = 9 frames, 144-dim each
  → 16 ConformerBlocks (or 2 for quick test)
encoder_out: (1, 9, 144)
```

### Step 2 — Decoder

Prepend SOS (blank = token 0):

```
decoder input = [0, 1, 2, 3, 4]   ← 5 tokens (SOS + 4 targets)
  → Embedding → (1, 5, 144)
  → LSTM
decoder_out : (1, 5, 320)
```

### Step 3 — Joint

Combine every encoder frame with every decoder state:

```
encoder: (1, 9, 144)
decoder: (1, 5, 320)
  → joint network
logits : (1, 9, 5, 29)   ← 9 frames × 5 decoder states × 29 vocab
```

### Step 4 — RNN-T Loss

The loss sums over all valid paths through the 9×5 grid that produce [1,2,3,4]:

```
log_probs = log_softmax(logits)   → (1, 9, 5, 29)
loss = RNNTLoss(log_probs, targets=[1,2,3,4], enc_len=[9], tgt_len=[4])
     = scalar (e.g. 96.2 at epoch 1, drops to 0.39 after training)
```

---

## Greedy Decoding (Inference)

At inference there is no loss. Instead the model emits tokens greedily:

```
For each encoder frame t:
    1. Get encoder_out[:, t, :]        ← audio at this frame
    2. Get decoder_out from last token  ← memory of what was said
    3. Joint network → logits over vocab
    4. Pick argmax
    5. If prediction != blank:
           emit the token
           feed it back to decoder as next input
```

This runs in real time, frame by frame, without needing the full audio upfront.

---

## Training Results (Toy Data, 20 Epochs)

| Epoch | Loss  | Sample Decode |
|-------|-------|---------------|
| 1     | 96.22 | (empty)       |
| 4     | 43.70 | 'o'           |
| 8     | 12.11 | 'attentionition...' |
| 12    | 3.42  | 'conformer model' |
| 20    | 0.39  | 'batch normalization' |

The model goes from outputting nothing to producing real words in 12 epochs on 100 synthetic samples.

---

## Paper Config (Table 1 — Conformer-S)

| Component | Value |
|---|---|
| Encoder d_model | 144 |
| Encoder layers | 16 |
| Attention heads | 4 |
| Conv kernel size | 32 |
| Decoder LSTM layers | 1 |
| Decoder dim | 320 |
| Joint dim | 320 |
| Total params | ~10.3M |
