# Convolution Module

**File:** `modules/convolution.py`

---

## What is it, in plain English?

If attention is like a person **reading the whole page** to understand context, convolution is like a person **carefully reading a few words at a time** with a magnifying glass.

The convolution module looks at a **small window of neighbouring frames** (kernel size = 32 frames in the paper, ~320ms of audio) and detects **local patterns** — things like the shape of a consonant, or the transition between phonemes.

> Attention = big picture. Convolution = local detail. Together = Conformer.

---

## Architecture (Figure 2 from paper)

```
Input (B, T, d)
  → LayerNorm
  → Pointwise Conv  (d → 2d)   ← expand channels
  → GLU                         ← gate: keeps only relevant half
  → 1-D Depthwise Conv          ← look at 32 neighbouring frames
  → BatchNorm
  → Swish
  → Pointwise Conv  (d → d)    ← project back
  → Dropout
Output (B, T, d)                ← caller adds residual: x'' = x' + Conv(x')
```

---

## Why "Depthwise"?

A **regular** conv on `d` channels with kernel `k` costs: `d × d × k` operations per position.

A **depthwise** conv treats each channel independently: `d × k` operations — much cheaper!

With d=144, k=32:
- Regular: 144 × 144 × 32 = **663,552 ops**
- Depthwise: 144 × 32 = **4,608 ops** ← **144× cheaper**

Each channel learns its own "local detector" without mixing channels. Channels are mixed by the pointwise convolutions before and after.

---

## GLU — Gated Linear Unit

The **GLU** is like a filter that decides what information to pass through.

You feed in a tensor of size `2d`, split it into two halves `A` and `B`:

$$\text{GLU}(A, B) = A \otimes \sigma(B)$$

- `A` = the content ("what I want to say")
- `σ(B)` = the gate ("how much to let through"), values between 0 and 1

**Example with d=2:**
```
Input [2d=4]: [3.0, 1.5, -1.2, 0.8]
Split:
  A = [3.0, 1.5]         ← content
  B = [-1.2, 0.8]        ← gate
  σ(B) = [σ(-1.2), σ(0.8)] = [0.23, 0.69]

GLU = A ⊗ σ(B) = [3.0×0.23, 1.5×0.69]
                = [0.69, 1.04]
```

The gate learned to suppress `A[0]` (only 23% passes through) but let `A[1]` mostly through (69%).

---

## Step-by-Step with [1, 2, 3, 4] (d=4 example)

**Input** (one batch, 4 frames, d=4):
```
x = [[1, 2, 3, 4],
     [0, 1, 0, 1],
     [2, 2, 2, 2],
     [4, 3, 2, 1]]
```

### Step 1 — LayerNorm
Normalise each frame to zero-mean, unit-variance.

### Step 2 — Transpose for Conv1d
Conv1d expects `(B, Channels, Time)` but we have `(B, Time, Channels)`:
```
Before transpose: (1, 4, 4)   ← (batch, time, dim)
After  transpose: (1, 4, 4)   ← (batch, dim, time)
```

### Step 3 — Pointwise Conv (d → 2d)
A 1×1 convolution expands channels from 4 to 8:
```
Shape: (1, 4, 4) → (1, 8, 4)
```
Think of it as projecting each time-step into a higher-dimensional space.

### Step 4 — GLU (2d → d)
Split 8 channels into two groups of 4:
```
A = first 4 channels   (content)
B = last  4 channels   (gate)
Output = A ⊗ sigmoid(B)   → shape (1, 4, 4)
```

### Step 5 — 1-D Depthwise Conv (kernel=3 in demo, 32 in paper)
Each of the 4 channels independently scans across the 4 time frames:
```
Channel 0 sees: [frame0_ch0, frame1_ch0, frame2_ch0, frame3_ch0]
                 and applies a 3-point sliding window
```
This is where **local temporal patterns** are captured. Shape stays `(1, 4, 4)`.

### Step 6 — BatchNorm + Swish
Normalise across the batch dimension, then apply Swish.

### Step 7 — Pointwise Conv (d → d)
Mix channels back together. Shape stays `(1, 4, 4)`.

### Step 8 — Transpose back + Residual
```
(1, 4, 4) → transpose → (1, 4, 4)   ← (batch, time, dim)
output = x + conv(x)
```

---

## Actual Output (from running the code)

```
After Pointwise  expand: (1, 16, 4)  ← d=8 expanded to 2d=16
After GLU              : (1, 8,  4)  ← gated back to d=8
After Depthwise Conv   : (1, 8,  4)  ← local patterns captured
After final projection : (1, 4,  8)  ← back to (batch, time, d)

Frame 0 input  = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
Frame 0 conv   = [0.236, 0.141, -0.159, -0.015, 0.183, -0.145, 0.241, -0.123]
Frame 0 output = [1.236, 2.141, 2.841, 3.985, 1.183, 1.855, 3.241, 3.877]
```

---

## Why kernel size = 32?

From the ablation study in the paper (Table 7):

| Kernel size | test-clean WER | test-other WER |
|---|---|---|
| 3  | 1.99 | 4.39 |
| 7  | 2.02 | 4.44 |
| 17 | 2.04 | 4.38 |
| **32** | **2.03** | **4.29** ✓ |
| 65 | 1.98 | 4.46 |

Kernel 32 = 320ms window. Long enough to capture full phoneme transitions, but not so long it becomes redundant with attention.
