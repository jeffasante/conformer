"""
train.py — Conformer-S Transducer training script
 
Follows the paper's exact training setup (Section 3.2):
  - Adam: beta1=0.9, beta2=0.98, eps=1e-9
  - Transformer LR schedule: warmup 10k steps, peak = 0.05 / sqrt(d)
  - L2 weight decay: 1e-6
  - Variational noise on weights
  - Dropout 0.1 throughout
 
Usage:
    python train.py                          # default 50 epochs
    python train.py --epochs 100             # more epochs
    python train.py --checkpoint model.pt    # resume from checkpoint
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
 
sys.path.insert(0, os.path.dirname(__file__))
 
from dataset import get_dataloader, VOCAB_SIZE, tokens_to_text
from conformer_transducer import ConformerTransducer
 

# LR SCHEDULE — exact formula from paper Section 3.2
class TransformerLRSchedule:
    """
    lr = (0.05 / sqrt(d_model)) * min(1/sqrt(step), step / warmup^1.5)
 
    This warms up linearly then decays as 1/sqrt(step).
    """
 
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 10_000):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0
        self._update_lr()
 
    def step(self):
        self.step_num += 1
        self._update_lr()
 
    def _update_lr(self):
        step = max(1, self.step_num)
        peak = 0.05 / math.sqrt(self.d_model)
        lr   = peak * min(
            1.0 / math.sqrt(step),
            step / (self.warmup_steps ** 1.5)
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
 
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# VARIATIONAL NOISE — paper Section 3.2
def apply_variational_noise(model: nn.Module, std: float = 1e-5):
    """
    Add small Gaussian noise to all weights each step.
    Helps regularise deep models (Graves 2012).
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * std)


# TRAINING LOOP
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice      : {device}")
    print(f"Vocab size  : {VOCAB_SIZE}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Samples     : {args.n_samples}")
 
    # data
    train_loader = get_dataloader(
        n_samples  = args.n_samples,
        batch_size = args.batch_size,
        shuffle    = True,
    )
 
    # model (Conformer-S, paper Table 1)
    model = ConformerTransducer(
        vocab_size  = VOCAB_SIZE,
        d_model     = 144,
        n_heads     = 4,
        n_layers    = args.n_layers,   # 16 for real, 4 for quick run
        decoder_dim = 320,
        joint_dim   = 320,
        kernel_size = args.kernel_size,
        dropout     = 0.1,
    ).to(device)
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total_params:,}\n")
 
    # optimiser — paper: Adam beta1=0.9 beta2=0.98 eps=1e-9
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = 1e-3,          # overridden immediately by scheduler
        betas        = (0.9, 0.98),
        eps          = 1e-9,
        weight_decay = 1e-6,          # L2 regularisation (paper Section 3.2)
    )
 
    # LR scheduler — transformer schedule with 10k warmup
    # For toy data we use scaled-down warmup so LR rises within our epochs
    '''
    So by doing this, the code forces the warmup
    to end after 25% of the total training time 
    if the dataset is small.

    This allows the model to actually reach the "decay" phase 
    where most of the fine-tuning happens, even on a tiny 
    laptop-sized dataset.
    '''
    warmup = min(args.warmup_steps, len(train_loader) * args.epochs // 4)
    scheduler = TransformerLRSchedule(optimizer, d_model=144, warmup_steps=warmup)
    print(f"Warmup steps: {warmup}")
    print("─" * 55)
    print(f"{'Epoch':>6}  {'Step':>6}  {'Loss':>8}  {'LR':>12}  {'Sample Decode'}")
    print("─" * 55)
 
    # resume from checkpoint
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt        = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        scheduler.step_num = ckpt.get("step_num", 0)
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")
 
    # training
    global_step = scheduler.step_num
    history     = []
 
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
 
        for batch_idx, (features, feat_len, targets, tgt_len) in enumerate(train_loader):
            features = features.to(device)
            feat_len = feat_len.to(device)
            targets  = targets.to(device)
            tgt_len  = tgt_len.to(device)
 
            optimizer.zero_grad()
 
            loss = model(features, feat_len, targets, tgt_len)
 
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] NaN/Inf loss at step {global_step} — skipping")
                continue
 
            loss.backward()
 
            # Gradient clipping — important for RNN-T stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
 
            # Variational noise (paper Section 3.2)
            apply_variational_noise(model, std=1e-5)
 
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss  += loss.item()
 
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
 
        # log every epoch
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            # Greedy decode one sample to show progress
            model.eval()
            sample_feat = features[:1].to(device)
            with torch.no_grad():
                decoded = model.greedy_decode(sample_feat, max_decode_len=30)
            decoded_text = tokens_to_text(decoded)
            model.train()
 
            print(
                f"{epoch+1:>6}  "
                f"{global_step:>6}  "
                f"{avg_loss:>8.4f}  "
                f"{scheduler.get_lr():>12.2e}  "
                f"'{decoded_text}'"
            )
 
    print("─" * 55)
    print(f"\nTraining complete. Final loss: {history[-1]:.4f}")
 
    # save checkpoint
    out_path = args.save_path
    torch.save({
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "epoch"     : args.epochs - 1,
        "step_num"  : global_step,
        "loss"      : history[-1],
        "vocab_size": VOCAB_SIZE,
        "config"    : {
            "d_model"    : 144,
            "n_heads"    : 4,
            "n_layers"   : args.n_layers,
            "decoder_dim": 320,
            "kernel_size": args.kernel_size,
        },
    }, out_path)
    print(f"Checkpoint saved to: {out_path}")
 
    # loss curve
    _print_loss_curve(history)
    return model, history
 
 
def _print_loss_curve(history: list[float]):
    """ASCII loss curve in terminal."""
    print("\nLoss curve:")
    if len(history) < 2:
        return
    lo, hi = min(history), max(history)
    rng    = max(hi - lo, 1e-6)
    height = 8
    width  = min(len(history), 60)
    step   = max(1, len(history) // width)
    vals   = history[::step][:width]
 
    for row in range(height, -1, -1):
        line = ""
        for v in vals:
            norm = (v - lo) / rng
            line += "█" if norm >= row / height else " "
        label = f"{lo + (hi-lo)*row/height:.3f} |" if row % 2 == 0 else "       |"
        print(f"  {label}{line}")
    print(f"  {'':7s} {'─'*len(vals)}")
    print(f"  {'':7s} epoch 1{' '*(len(vals)-16)}epoch {len(history)}")
 
 
# MAIN
def main():
    parser = argparse.ArgumentParser(description="Train Conformer-S Transducer")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--n_samples",    type=int,   default=100)
    parser.add_argument("--n_layers",     type=int,   default=4,
                        help="Encoder layers (16 for full model, 4 for quick test)")
    parser.add_argument("--kernel_size",  type=int,   default=15,
                        help="Conv kernel size (32 paper, 15 for short sequences)")
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--log_every",    type=int,   default=5)
    parser.add_argument("--save_path",    type=str,   default="conformer_s.pt")
    parser.add_argument("--checkpoint",   type=str,   default=None)
    args = parser.parse_args()
    train(args)
 
 
if __name__ == "__main__":
    main()
 