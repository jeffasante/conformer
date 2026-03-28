# Training Script

**File:** `modules/train.py`

---

## What is it, in plain English?

The training script is the engine that runs all the other pieces together repeatedly until the model learns.

Each pass through the data:
1. Feed audio features into the model
2. Compare the model's output to the correct transcript
3. Measure how wrong it was (the loss)
4. Work backwards through the network adjusting every weight slightly (backpropagation)
5. Repeat

After enough repetitions the model gets better at predicting transcripts from audio.

---

## The Training Loop

```
For each epoch:
    For each batch:
        1. Forward pass  → compute loss
        2. Backward pass → compute gradients
        3. Clip gradients
        4. Add variational noise
        5. Update weights (optimizer step)
        6. Update learning rate (scheduler step)
    Log average loss for this epoch
Save checkpoint
```

---

## Paper Training Settings (Section 3.2)

| Setting | Paper Value | This Script |
|---|---|---|
| Optimizer | Adam | Adam |
| beta1 | 0.9 | 0.9 |
| beta2 | 0.98 | 0.98 |
| epsilon | 1e-9 | 1e-9 |
| L2 weight decay | 1e-6 | 1e-6 |
| Warmup steps | 10,000 | scaled to data size |
| Peak LR formula | 0.05 / sqrt(d) | 0.05 / sqrt(144) |
| Variational noise | yes | yes (std=1e-5) |
| Grad clip | not stated | 1.0 (standard practice) |

---

## The Learning Rate Schedule — Step by Step

The paper uses a transformer learning rate schedule. The formula:

$$lr = \frac{0.05}{\sqrt{d}} \times \min\!\left(\frac{1}{\sqrt{\text{step}}}, \frac{\text{step}}{10000^{1.5}}\right)$$

With d = 144 (Conformer-S):

```
peak = 0.05 / sqrt(144) = 0.05 / 12 = 0.00417
```

### Phase 1 — Warmup (steps 1 to warmup_steps)

During warmup, the second term in the min() is smaller, so it controls:

```
lr = peak × step / warmup^1.5
```

With warmup = 500 and step = 1, 100, 500:

```
step   1: lr = 0.00417 × 1   / 500^1.5 = 0.00000037
step 100: lr = 0.00417 × 100 / 500^1.5 = 0.000037
step 500: lr = 0.00417 × 500 / 500^1.5 = 0.000186  ← peak reached
```

The LR rises linearly from near zero to the peak.

### Phase 2 — Decay (steps > warmup_steps)

After warmup, the first term takes over:

```
lr = peak × 1 / sqrt(step)
```

```
step  500: lr = 0.00417 / sqrt(500)  = 0.000186
step 1000: lr = 0.00417 / sqrt(1000) = 0.000132
step 2000: lr = 0.00417 / sqrt(2000) = 0.000093
```

The LR decays slowly. This prevents overshooting the loss minimum in later training.

### Why warmup?

At the start of training the model is random and the gradients are noisy. A high LR on random weights causes the loss to explode. Warmup keeps the LR small while the model stabilises, then increases it once gradients become more meaningful.

---

## Variational Noise — Plain English

After each weight update, tiny random noise is added to every parameter:

```python
for each weight w:
    w = w + noise     where noise ~ Normal(0, 1e-5)
```

This is a form of regularisation. It slightly blurs the weights, preventing the model from memorising exact patterns in the training data. For 100 toy samples it does not matter much, but for real data it helps generalisation.

---

## Gradient Clipping — Plain English

During backpropagation, gradients can sometimes become very large (exploding gradients), especially with RNN-T loss and LSTMs. This would cause the weights to jump wildly and destabilise training.

Gradient clipping rescales all gradients so their total norm never exceeds 1.0:

```
if ||gradients|| > 1.0:
    gradients = gradients × (1.0 / ||gradients||)
```

With our sample batch:

```
gradients before clip: [0.3, 4.7, 0.1, 2.1, ...]   norm = 5.4
scaling factor       : 1.0 / 5.4 = 0.185
gradients after clip : [0.056, 0.870, 0.019, 0.389, ...]   norm = 1.0
```

The direction is preserved, only the magnitude is capped.

---

## What the Loss Numbers Mean

RNN-T loss is negative log probability. Lower is better.

```
Loss = 96  →  probability of correct transcript ≈ e^{-96} ≈ 0  (random model)
Loss = 10  →  probability ≈ e^{-10} ≈ 0.000045  (learning something)
Loss =  1  →  probability ≈ e^{-1}  ≈ 0.37      (doing well)
Loss =  0.4 → probability ≈ e^{-0.4} ≈ 0.67     (good fit on toy data)
```

On our 100 synthetic samples the loss went from 96.2 to 0.39 in 20 epochs. On real data, training for hundreds of epochs on thousands of hours would push WER below 5%.

---

## Actual Training Output

```
Epoch   Step      Loss            LR  Sample Decode
      1     13   96.2206      1.53e-04  ''
      4     52   43.7019      5.78e-04  'o'
      8    104   12.1069      4.09e-04  'attentionitionition'
     12    156    3.4214      3.34e-04  'conformer model'
     20    260    0.3898      2.58e-04  'batch normalization'
```

The model starts outputting nothing, then partial words, then real phrases as the loss drops.

---

## Resuming From a Checkpoint

The script saves everything needed to resume:

```python
torch.save({
    "model"     : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    "epoch"     : current_epoch,
    "step_num"  : global_step,
    "loss"      : final_loss,
    "config"    : { d_model, n_heads, n_layers, ... }
}, "conformer_s.pt")
```

To resume:
```bash
python train.py --checkpoint conformer_s.pt --epochs 100
```

The optimizer state includes momentum buffers so the Adam optimiser continues exactly where it left off rather than resetting.

---

## Running Options

```bash
# Quick verification (2 layers, small kernel, 20 epochs)
python train.py --epochs 20 --n_layers 2 --kernel_size 3

# Full Conformer-S as in the paper (needs GPU)
python train.py --epochs 500 --n_layers 16 --kernel_size 32 --warmup_steps 10000

# Resume from checkpoint
python train.py --checkpoint conformer_s.pt --epochs 200
```
