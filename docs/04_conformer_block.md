# Conformer Block

**File:** `modules/conformer_block.py`

---

## What is it, in plain English?

The Conformer Block is the **main repeating unit** — like one floor of a building. The full encoder stacks 16 or 17 of these floors on top of each other.

Each block takes an audio representation in, refines it using all four sub-modules, and passes the improved representation to the next block.

The pattern is called **"Macaron"** because — like the French sandwich cookie — it has the same layer on both the outside (FFN), with different fillings in the middle (Attention + Conv).

---

## The Equation (from the paper, Section 2.4)

$$\tilde{x}_i = x_i + \frac{1}{2}\text{FFN}(x_i)$$

$$x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$$

$$x''_i = x'_i + \text{Conv}(x'_i)$$

$$y_i = \text{LayerNorm}\!\left(x''_i + \frac{1}{2}\text{FFN}(x''_i)\right)$$

In plain words:

1. **Refine a little** (half-FFN)
2. **Look at the global context** (attention)
3. **Look at local patterns** (convolution)
4. **Refine again and stabilise** (half-FFN + LayerNorm)

---

## Why This Specific Order?

The paper tested different arrangements (Table 4). Key findings:

| Arrangement | dev-other WER |
|---|---|
| **FFN → MHSA → Conv → FFN** (Conformer) | **4.4** ✓ |
| Conv before MHSA | 4.5 |
| MHSA and Conv in parallel | 4.9 |

Convolution **after** attention works best because:
- Attention first identifies **which frames matter globally**
- Convolution then refines **local patterns** within that global context

---

## Full Walkthrough with [1, 2, 3, 4]

We'll trace one audio frame `[1, 2, 3, 4, 1, 2, 3, 4]` (d=8) through all 4 steps.

---

### Step 1: First Half-FFN

The FFN makes a small learned correction:
```
x           = [1.000, 2.000, 3.000, 4.000, 1.000, 2.000, 3.000, 4.000]
0.5×FFN(x)  = [0.009, 0.136, -0.080, 0.117, -0.034, 0.056, -0.073, -0.070]
──────────────────────────────
After FFN-1 = [1.009, 2.136, 2.920, 4.117, 0.967, 2.056, 2.927, 3.931]
```

Small nudges — the signal is mostly preserved.

---

### Step 2: Multi-Head Self-Attention

Now the frame can "look at" all 4 frames and absorb context:
```
After FFN-1  = [1.009, 2.136, 2.920, 4.117, 0.967, 2.056, 2.927, 3.931]
MHSA output  = [-0.210, -0.625, -0.278, 0.339, -0.470, 0.266, 0.244, 0.032]
──────────────────────────────
After MHSA   = [0.800, 1.511, 2.644, 4.456, 0.497, 2.322, 3.171, 3.963]
```

Some values shift notably — the frame has now "heard from" its neighbours.

---

### Step 3: Convolution Module

Local patterns in time are now captured:
```
After MHSA  = [0.800, 1.511, 2.644, 4.456, 0.497, 2.322, 3.171, 3.963]
Conv output = [0.199, -0.301, 0.139, -0.020, 0.153, 0.142, 0.155, -0.074]
──────────────────────────────
After Conv  = [0.999, 1.211, 2.784, 4.435, 0.650, 2.464, 3.326, 3.889]
```

---

### Step 4: Second Half-FFN + LayerNorm

Final refinement, then normalise to stabilise the signal:
```
After Conv   = [0.999, 1.211, 2.784, 4.435, 0.650, 2.464, 3.326, 3.889]
0.5×FFN(x'') ≈ small corrections again
LayerNorm    → zero mean, unit variance
──────────────────────────────
Final output = [-1.071, -1.029, 0.224, 1.500, -1.371, -0.002, 0.666, 1.082]
```

Note how the output is now **centred around 0** — that's LayerNorm at work. This keeps gradients healthy for training.

---

## Actual Output (from running the code)

```
Input     : [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
→ FFN-1   : [1.009, 2.136, 2.920, 4.117, 0.967, 2.056, 2.927, 3.931]
→ MHSA    : [0.800, 1.511, 2.644, 4.456, 0.497, 2.322, 3.171, 3.963]
→ Conv    : [0.999, 1.211, 2.784, 4.435, 0.650, 2.464, 3.326, 3.889]
→ FFN-2+LN: [-1.071, -1.029, 0.224, 1.500, -1.371, -0.002, 0.666, 1.082]
```

---

## What Does Each Step Contribute?

| Step | What it adds |
|------|-------------|
| FFN-1 | Feature transformation — learns non-linear combinations |
| MHSA  | Global context — connects distant frames |
| Conv  | Local context — captures phoneme-level patterns |
| FFN-2 | Further refinement of combined representation |
| LayerNorm | Stability — prevents exploding/vanishing gradients |

---

## Why Stack 16 Blocks?

Each block learns a slightly different view of the audio. Earlier blocks might learn basic acoustic features (pitch, energy), while later blocks learn higher-level patterns (phonemes, syllables, word boundaries).

It's like passing the audio through 16 different expert listeners, each one building on what the previous one noticed.
