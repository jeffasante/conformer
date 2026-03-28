# RNN-T Decoder (Prediction Network)

**File:** `modules/decoder.py`

---

## What is it, in plain English?

The decoder is the part of the model that keeps track of **what has already been said**.

Think of it like a person transcribing audio who remembers the last word they wrote. That memory helps them predict the next word. The decoder holds that memory.

In RNN-T, the decoder never sees the audio directly. It only sees the **previously predicted tokens** and produces a hidden state that gets combined with the audio encoding later in the joint network.

---

## Architecture

```
Previous token (integer index)
  → Embedding layer     (integer → vector)
  → Single LSTM layer   (vector → hidden state)
Output: hidden state (B, U, decoder_dim)
```

The paper (Section 3.2, Table 1) uses:
- 1 LSTM layer
- decoder dim = 320 for Conformer-S
- decoder dim = 640 for Conformer-M and L

---

## What is an Embedding?

Before the LSTM can process a token, the integer index needs to become a vector.

An embedding is just a lookup table. Each token gets its own row.

With our sample data, vocab = 29 tokens (a-z, space, apostrophe, blank):

```
Token index 1  →  [0.12, -0.34, 0.56, ..., 0.09]   ← 144 numbers for 'a'
Token index 2  →  [-0.23, 0.45, -0.11, ..., 0.77]  ← 144 numbers for 'b'
...
```

These 144 numbers are learned during training. At the start they are random.

---

## What is an LSTM?

LSTM stands for Long Short-Term Memory. It is a type of recurrent network that processes sequences one step at a time while carrying a memory (called the hidden state) forward.

At each step it asks three questions:
- What from the past should I forget?
- What new information should I remember?
- What should I output right now?

These are controlled by learned gates (forget gate, input gate, output gate).

---

## Step-by-Step with [1, 2, 3, 4]

Let our sample tokens be the indices for the letters "a", "b", "c", "d":

```
targets = [1, 2, 3, 4]   ← token indices for 'a', 'b', 'c', 'd'
```

### Step 1 — Embedding lookup

Each index is replaced by its learned vector (dimension = 144):

```
1 → [0.12, -0.34,  0.56,  0.09, ...]   ← embedding for 'a'
2 → [-0.23, 0.45, -0.11,  0.77, ...]   ← embedding for 'b'
3 → [0.88,  0.02,  0.34, -0.55, ...]   ← embedding for 'c'
4 → [-0.41, 0.67,  0.23,  0.14, ...]   ← embedding for 'd'
```

Shape after embedding: (1, 4, 144) — batch=1, 4 tokens, 144-dim each.

### Step 2 — LSTM processes the sequence

The LSTM reads the embeddings one token at a time and updates its hidden state:

```
Step 1: reads 'a' embedding  →  hidden state h1 (320-dim)
Step 2: reads 'b' embedding  +  h1  →  hidden state h2
Step 3: reads 'c' embedding  +  h2  →  hidden state h3
Step 4: reads 'd' embedding  +  h3  →  hidden state h4
```

At each step the LSTM combines the current embedding with the previous hidden state through its gates.

### Output

The LSTM outputs a hidden state at every step:

```
h1 = [0.03, -0.12, 0.44, ...]   ← after seeing 'a'
h2 = [0.11,  0.08, 0.39, ...]   ← after seeing 'a', 'b'
h3 = [-0.05, 0.21, 0.47, ...]   ← after seeing 'a', 'b', 'c'
h4 = [0.18, -0.03, 0.52, ...]   ← after seeing 'a', 'b', 'c', 'd'
```

Shape: (1, 4, 320) — batch=1, 4 time steps, 320-dim hidden state.

---

## Why Prepend SOS (Start of Sequence)?

During training the decoder input is shifted by one:

```
Actual transcript:   [a, b, c, d]
Decoder input:       [blank, a, b, c, d]   ← SOS prepended
```

This is so that when the model is predicting token position 1, it has only seen the blank (start), not 'a' itself. It forces the model to actually predict rather than just copy.

---

## Actual Output (from running the code)

```
Input  (2 samples, 5 tokens): torch.Size([2, 5])
Output                       : torch.Size([2, 5, 320])
```

Each of the 5 positions now has a 320-dimensional vector encoding the history of tokens seen so far.
