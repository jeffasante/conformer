# Multi-Head Self-Attention Module (MHSA)

**File:** `modules/attention.py`

---

## What is it, in plain English?

Self-attention lets every audio frame **look at every other frame** and ask:
> *"Which other parts of this utterance are relevant to me right now?"*

For example, when recognising the word "Ghana", the model might learn that the `/a/` sound at the end is related to the `/G/` at the beginning — even if they're far apart in time.

**Multi-head** means doing this several times simultaneously, each "head" learning to attend to **different kinds of relationships** (one might focus on nearby sounds, another on long-range dependencies).

---

## Architecture (Figure 3 from paper)

```
Input (B, T, d)
  → LayerNorm          ← pre-norm (applied BEFORE attention)
  → Split into Q, K, V
  → Add Relative Positional Encoding to K
  → Scaled Dot-Product Attention × H heads
  → Concat heads → Linear projection
  → Dropout
Output (B, T, d)       ← caller adds residual: x' = x + MHSA(x)
```

---

## Why Relative Positional Encoding?

Standard Transformers use **absolute** positional encoding: frame 1, frame 2, frame 3...

The problem: a short sentence and a long sentence use the same absolute positions, but the **relative distance** between sounds matters more in speech.

Relative encoding says: *"frame i is 3 steps before frame j"* rather than *"frame i is at position 7"*. This makes the model generalise better to utterances of different lengths.

---

## The Attention Calculation — Step by Step

### Setup: 4 frames, dimension d=4, 2 heads → d_k = d/heads = 2

**Input** (4 audio frames, each 4-dimensional):
```
x = [[1, 2, 3, 4],    ← frame 0
     [0, 1, 0, 1],    ← frame 1
     [2, 2, 2, 2],    ← frame 2
     [4, 3, 2, 1]]    ← frame 3
```

---

### Step 1 — Project to Q, K, V

Three learned weight matrices turn each frame into:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**:   "What do I offer?"
- **V (Value)**: "What's my actual content?"

With simplified weights (identity for illustration):
```
Q = K = V = x   (just the input, for this example)
```

---

### Step 2 — Compute Attention Scores

For each pair of frames (i, j), compute how much frame i should "attend to" frame j:

$$\text{score}(i, j) = \frac{Q_i \cdot K_j}{\sqrt{d_k}}$$

Scale factor $\sqrt{d_k} = \sqrt{2} \approx 1.41$ prevents scores from getting too large.

**Example: How much does frame 0 attend to frame 1?**
```
Q_0 = [1, 2]    (first half of frame 0 after splitting for head 1)
K_1 = [0, 1]

score = (1×0 + 2×1) / √2 = 2 / 1.41 = 1.41
```

Do this for all pairs → 4×4 score matrix:
```
scores =
         f0    f1    f2    f3
frame 0 [2.50  0.71  2.83  2.83]
frame 1 [0.71  0.71  1.41  0.71]
frame 2 [2.83  1.41  2.83  2.83]
frame 3 [2.83  0.71  2.83  3.54]
```

---

### Step 3 — Softmax → Attention Weights

Apply softmax to each row so weights sum to 1:

$$\text{attn}(i, j) = \frac{e^{\text{score}(i,j)}}{\sum_k e^{\text{score}(i,k)}}$$

**Frame 0's attention weights** (after softmax):
```
scores  = [2.50,  0.71,  2.83,  2.83]
softmax → [0.22,  0.03,  0.37,  0.37]
```

Interpretation: Frame 0 pays 37% attention to frame 2, 37% to frame 3, 22% to itself, and barely any to frame 1.

---

### Step 4 — Weighted Sum of Values

$$\text{output}_i = \sum_j \text{attn}(i,j) \cdot V_j$$

**Frame 0's output** (simplified):
```
= 0.22 × V_0 + 0.03 × V_1 + 0.37 × V_2 + 0.37 × V_3
= 0.22×[1,2,3,4] + 0.03×[0,1,0,1] + 0.37×[2,2,2,2] + 0.37×[4,3,2,1]
= [0.22,0.44,...] + [0,0.03,...] + [0.74,0.74,...] + [1.48,1.11,...]
≈ [2.44, 2.32, 2.18, 2.03]
```

Frame 0's output is now a **blend of all frames**, weighted by relevance.

---

### Step 5 — Multi-Head: Do it H times, then concatenate

With H=2 heads (each with d_k=2):
```
head_1 output: [2.44, 2.32]
head_2 output: [1.81, 1.94]
concat:        [2.44, 2.32, 1.81, 1.94]  → shape d=4 ✓
```

Final linear projection → back to d=4.

---

## Actual Output (from running the code)

```
Frame 0 input  = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
Frame 0 MHSA   = [0.151, -0.139, -0.382, 0.088, -0.191, 0.509, -0.322, -0.035]
Frame 0 output = [1.151, 1.861, 2.618, 4.088, 0.809, 2.509, 2.679, 3.966]
```

---

## Why is This Powerful for Speech?

| Problem | How MHSA Solves It |
|---|---|
| Long-range dependencies | Every frame can directly attend to any other frame |
| Variable-length audio | Relative positional encoding adapts to any length |
| Ambiguous sounds | Attending to surrounding context resolves ambiguity |
| Different linguistic patterns | Multiple heads learn different relationship types |
