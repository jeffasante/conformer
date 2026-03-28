# RNN-T Joint Network

**File:** `modules/joint.py`

---

## What is it, in plain English?

The joint network is the **meeting point** of audio and text.

At every combination of (audio frame, decoded token), the joint network asks:

> "Given what the audio sounds like at this moment, and given what tokens I have predicted so far, what is the next most likely token?"

It is a small neural network that takes one audio vector and one decoder vector, adds them together, and turns the result into a probability over every possible token.

---

## Why Does It Produce a 4D Output?

The encoder produces T audio frames. The decoder produces U+1 token states. The joint network evaluates every combination:

```
T frames × (U+1) token states = T × (U+1) combinations
```

For each combination it outputs a probability distribution over the vocabulary.

That is why the output shape is (B, T, U+1, vocab_size).

The RNN-T loss then finds the most likely alignment path through this grid.

---

## Architecture

```
encoder_out  (B, T,   encoder_dim)
decoder_out  (B, U+1, decoder_dim)

  → Linear: encoder_dim → joint_dim     ← project encoder
  → Linear: decoder_dim → joint_dim     ← project decoder
  → Add both projections
  → Tanh
  → Linear: joint_dim → vocab_size
  → log_softmax
Output: (B, T, U+1, vocab_size)
```

---

## Step-by-Step with [1, 2, 3, 4]

Let's use tiny numbers to make the maths clear.

Say:
- joint_dim = 4
- vocab_size = 5 (blank, a, b, c, d)
- T = 2 audio frames
- U+1 = 3 decoder states

### Input vectors (simplified to 4-dim for clarity)

One audio frame from the encoder:
```
enc_frame_0 = [1.0, 2.0, 3.0, 4.0]
```

One decoder state:
```
dec_state_0 = [0.5, 1.0, 1.5, 2.0]
```

### Step 1 — Project both to joint_dim

With simplified weight matrices (identity for illustration):

```
enc_projected = W_enc × enc_frame_0 = [1.0, 2.0, 3.0, 4.0]
dec_projected = W_dec × dec_state_0 = [0.5, 1.0, 1.5, 2.0]
```

### Step 2 — Add them together

```
combined = enc_projected + dec_projected
         = [1.0+0.5, 2.0+1.0, 3.0+1.5, 4.0+2.0]
         = [1.5, 3.0, 4.5, 6.0]
```

### Step 3 — Tanh activation

Tanh squishes values to the range (-1, 1):

```
Tanh([1.5, 3.0, 4.5, 6.0])
   = [0.905, 0.995, 0.9998, 0.9999]
```

The larger values saturate near 1. This prevents the combined signal from exploding.

### Step 4 — Project to vocab_size

A final linear layer maps the 4-dim joint vector to 5 logits (one per token):

```
logits = W_out × [0.905, 0.995, 0.9998, 0.9999]
       = [0.21, -0.14, 0.88, 0.33, -0.55]
         (blank,  a,    b,    c,    d)
```

### Step 5 — Log-softmax

Convert logits to log-probabilities:

```
softmax([0.21, -0.14, 0.88, 0.33, -0.55])
      = [0.17,  0.12, 0.33, 0.20,  0.08]   ← probabilities sum to 1

log:    [-1.77, -2.12, -1.11, -1.61, -2.53]
```

At this (frame, decoder_state) combination, token 'b' (index 2) has the highest probability of 0.33.

### Full output shape

This computation happens for all T×(U+1) combinations simultaneously:

```
T=2, U+1=3 → 6 combinations, each producing 5 log-probs
Output shape: (1, 2, 3, 5)
```

---

## Actual Output (from running the code)

```
Encoder input : (2, 10, 144)   ← 2 samples, 10 frames, 144-dim
Decoder input : (2,  6, 320)   ← 2 samples, 6 decoder states
Logits output : (2, 10,  6, 30)  ← every frame × every state × 30 tokens
```

---

## What the RNN-T Loss Does With This

The RNN-T loss receives the full (B, T, U+1, vocab_size) grid and finds the total probability of all valid alignments that produce the correct transcript.

A valid alignment is any path through the grid that, after removing blank tokens, spells out the target text. There can be many such paths (the blank can appear anywhere).

The loss is the negative log of the sum of all valid alignment probabilities. Training pushes this higher, meaning the model assigns more probability to correct transcripts.
