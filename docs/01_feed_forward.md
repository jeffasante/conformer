# Feed Forward Module (FFN)

**File:** `modules/feed_forward.py`

---

## What is it, in plain English?

Think of the Feed Forward Module as a **"thinking layer"** that takes each audio frame and asks:
> *"Given what I see here, what's important? Let me expand my thoughts, filter them, then compress back."*

It does **nothing** with neighbouring frames — it only looks at **one frame at a time**. All 4 frames get processed in parallel.

---

## Architecture (Figure 4 from paper)

```
Input (d)
  → LayerNorm
  → Linear: d → 4d      ← expand (think: "brainstorm 4× as many ideas")
  → Swish activation
  → Dropout
  → Linear: 4d → d      ← compress back
  → Dropout
Output (d)
```

---

## Swish Activation

Instead of the usual ReLU (just clips negatives to 0), the paper uses **Swish**:

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

With our sample data `x = [1, 2, 3, 4]`:

| x | σ(x) | Swish(x) = x·σ(x) |
|---|------|-------------------|
| 1 | 0.731 | **0.731** |
| 2 | 0.880 | **1.762** |
| 3 | 0.952 | **2.856** |
| 4 | 0.982 | **3.928** |

Swish is smoother than ReLU and was found to speed up convergence in Conformer.

---

## The Half-Step Residual — Why ½?

The Conformer uses FFN **twice** (Macaron-Net style). Each one contributes only **half** its output to the residual:

$$\tilde{x} = x + \frac{1}{2} \cdot \text{FFN}(x)$$

**Why?** Because you're stacking two FFNs. If both used full residuals, the signal would be amplified too much. Using ½ means together they add up to roughly one full FFN update — more stable training.

Think of it like two people each giving you **50% of the answer** rather than one person giving 100%.

---

## Worked Example with [1, 2, 3, 4]

Let's use a simplified FFN with dimension `d=4` (expansion to `4d=16`).

**Input vector** (one audio frame):
```
x = [1, 2, 3, 4]
```

**Step 1 — LayerNorm** (zero-mean, unit-variance):
```
mean   = (1+2+3+4)/4 = 2.5
std    = sqrt(((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²)/4) = 1.118
normed = [-1.342, -0.447, 0.447, 1.342]
```

**Step 2 — Linear expand** (d→4d, simplified with identity weights):
```
expanded = normed repeated 4× → shape [16]
```

**Step 3 — Swish** (applied elementwise):
```
Swish([-1.342, ...]) → values slightly smoothed around 0
```

**Step 4 — Linear compress** (4d→d):
```
compressed → back to shape [4]
```

**Step 5 — Half-step residual**:
```
ffn_output ≈ [0.017, 0.272, -0.159, 0.233]   (from actual run)
result = x + 0.5 * ffn_output
       = [1, 2, 3, 4] + [0.009, 0.136, -0.080, 0.117]
       = [1.009, 2.136, 2.920, 4.117]
```

The values barely change — the residual keeps the original signal intact while adding a **small learned correction**.

---

## Actual Output (from running the code)

```
Input  x      = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
FFN output     = [0.017, 0.272, -0.159, 0.233, -0.067, 0.113, -0.147, -0.139]
After residual = [1.009, 2.136, 2.920, 4.117, 0.967, 2.056, 2.927, 3.931]
```

Notice the output is close to the input — the FFN makes **subtle refinements**, it doesn't transform wildly.

---

## Parameter Count (Conformer-S, d=144)

| Layer | Params |
|-------|--------|
| Linear 1 (144→576) | 144×576 + 576 = 83,520 |
| Linear 2 (576→144) | 576×144 + 144 = 83,088 |
| **Total per FFN** | **~166k** |
| × 2 FFNs per block | ~332k |
| × 16 blocks | **~5.3M** |

FFNs account for the majority of the model's parameters!
