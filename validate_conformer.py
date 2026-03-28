"""
validate_conformer.py  —  fixed version

Two bugs fixed from v1:
  1. Prefix was doubling:  encoder.layers.0.0  →  now correctly encoder.layers.0
  2. Attention key names:  q_proj/k_proj       →  now linear_q/linear_k (NeMo naming)

Also handles NeMo's extra keys (linear_pos, pos_bias_u/v) gracefully —
those belong to the full Transformer-XL RPE which your simplified
attention module doesn't implement. They are skipped with a note.

Usage:
    python validate_conformer.py --nemo_path stt_en_conformer_ctc_small.nemo
"""

import argparse
import zipfile
import tarfile
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from feed_forward import FeedForwardModule
from attention import MultiHeadSelfAttentionModule
from convolution import ConvolutionModule
from conformer_block import ConformerBlock


# STEP 1 — inspect


def inspect_nemo(nemo_path: str):
    print("\n" + "═" * 60)
    print(f"STEP 1 — Contents of {os.path.basename(nemo_path)}")
    print("═" * 60)

    if nemo_path.endswith(".ckpt") or nemo_path.endswith(".pt"):
        print(f"  (Raw checkpoint file detected, skipping extraction)")
        return

    if zipfile.is_zipfile(nemo_path):
        with zipfile.ZipFile(nemo_path, "r") as z:
            for name in z.namelist():
                print(" ", name)
    else:
        try:
            with tarfile.open(nemo_path, "r:gz") as t:
                for name in t.getnames():
                    print(" ", name)
        except tarfile.ReadError:
            with tarfile.open(nemo_path, "r") as t:
                for name in t.getnames():
                    print(" ", name)


# STEP 2 — load weights


def load_nemo_weights(nemo_path: str, extract_dir: str = "/tmp/nemo_extracted"):
    print("\n" + "═" * 60)
    print("STEP 2 — Loading NeMo weights")
    print("═" * 60)

    if nemo_path.endswith(".ckpt") or nemo_path.endswith(".pt"):
        weights_file = nemo_path
    else:
        os.makedirs(extract_dir, exist_ok=True)
        # (existing zip/tar extraction logic follows...)

        if zipfile.is_zipfile(nemo_path):
            with zipfile.ZipFile(nemo_path, "r") as z:
                z.extractall(extract_dir)
        else:
            try:
                with tarfile.open(nemo_path, "r:gz") as t:
                    t.extractall(extract_dir)
            except tarfile.ReadError:
                with tarfile.open(nemo_path, "r") as t:
                    t.extractall(extract_dir)

        weights_file = None
        for fname in os.listdir(extract_dir):
            if fname.endswith(".ckpt") or fname.endswith(".pt"):
                weights_file = os.path.join(extract_dir, fname)
                break

    if weights_file is None:
        raise FileNotFoundError(
            f"No .ckpt or .pt found in {extract_dir}. Files: {os.listdir(extract_dir)}"
        )

    print(f"  Found: {weights_file}")
    state_dict = torch.load(weights_file, map_location="cpu", weights_only=False)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    print(f"  Total keys: {len(state_dict)}")
    return state_dict


# STEP 3 — print keys for one block


def print_key_map(state_dict: dict, block_idx: int = 0):
    print("\n" + "═" * 60)
    print(f"STEP 3 — NeMo keys for block {block_idx}")
    print("═" * 60)

    prefix = f"encoder.layers.{block_idx}"
    block_keys = {k: v.shape for k, v in state_dict.items() if prefix + "." in k}

    # If not found, look for ANY key that contains the pattern
    if not block_keys:
        for k in state_dict.keys():
            if f"layers.{block_idx}." in k:
                # auto-detect prefix: take everything before the block_idx
                # e.g. "1.encoder.layers.0.something" -> "1.encoder.layers.0"
                parts = k.split(f"layers.{block_idx}.")
                prefix = parts[0] + f"layers.{block_idx}"
                block_keys = {
                    pk: pv.shape
                    for pk, pv in state_dict.items()
                    if pk.startswith(prefix + ".")
                }
                break

    if not block_keys:
        print("  Could not find block keys. Printing first 40 keys:")
        for k, v in list(state_dict.items())[:40]:
            print(f"  {k:70s} {str(v)}")
        return None, None

    for k, v in block_keys.items():
        print(f"  {k:70s} {str(v)}")

    return block_keys, prefix


# STEP 4 — weight map  (NeMo → your module)


def build_weight_map(prefix: str) -> dict:
    """
    prefix is already the full block prefix e.g. "encoder.layers.0"
    Do NOT append block_idx again — that was the v1 bug.

    NeMo uses:  linear_q / linear_k / linear_v / linear_out
    Your code:  q_proj   / k_proj   / v_proj   / out_proj

    NeMo also has linear_pos, pos_bias_u, pos_bias_v (full Transformer-XL RPE).
    These are NOT mapped — your simplified attention doesn't have them.
    """
    p = prefix

    mapping = {
        # ── FFN 1 ──────────
        "ffn1.norm.weight": f"{p}.norm_feed_forward1.weight",
        "ffn1.norm.bias": f"{p}.norm_feed_forward1.bias",
        "ffn1.linear1.weight": f"{p}.feed_forward1.linear1.weight",
        "ffn1.linear1.bias": f"{p}.feed_forward1.linear1.bias",
        "ffn1.linear2.weight": f"{p}.feed_forward1.linear2.weight",
        "ffn1.linear2.bias": f"{p}.feed_forward1.linear2.bias",
        # ── MHSA  (NeMo: linear_q not q_proj) ────────────────────────────
        "mhsa.norm.weight": f"{p}.norm_self_att.weight",
        "mhsa.norm.bias": f"{p}.norm_self_att.bias",
        "mhsa.q_proj.weight": f"{p}.self_attn.linear_q.weight",
        "mhsa.q_proj.bias": f"{p}.self_attn.linear_q.bias",
        "mhsa.k_proj.weight": f"{p}.self_attn.linear_k.weight",
        "mhsa.k_proj.bias": f"{p}.self_attn.linear_k.bias",
        "mhsa.v_proj.weight": f"{p}.self_attn.linear_v.weight",
        "mhsa.v_proj.bias": f"{p}.self_attn.linear_v.bias",
        "mhsa.out_proj.weight": f"{p}.self_attn.linear_out.weight",
        "mhsa.out_proj.bias": f"{p}.self_attn.linear_out.bias",
        # ── Convolution ────
        "conv.norm.weight": f"{p}.norm_conv.weight",
        "conv.norm.bias": f"{p}.norm_conv.bias",
        "conv.pw_conv1.weight": f"{p}.conv.pointwise_conv1.weight",
        "conv.pw_conv1.bias": f"{p}.conv.pointwise_conv1.bias",
        "conv.dw_conv.weight": f"{p}.conv.depthwise_conv.weight",
        "conv.dw_conv.bias": f"{p}.conv.depthwise_conv.bias",
        "conv.bn.weight": f"{p}.conv.batch_norm.weight",
        "conv.bn.bias": f"{p}.conv.batch_norm.bias",
        "conv.bn.running_mean": f"{p}.conv.batch_norm.running_mean",
        "conv.bn.running_var": f"{p}.conv.batch_norm.running_var",
        "conv.pw_conv2.weight": f"{p}.conv.pointwise_conv2.weight",
        "conv.pw_conv2.bias": f"{p}.conv.pointwise_conv2.bias",
        # ── FFN 2 ──────────
        "ffn2.norm.weight": f"{p}.norm_feed_forward2.weight",
        "ffn2.norm.bias": f"{p}.norm_feed_forward2.bias",
        "ffn2.linear1.weight": f"{p}.feed_forward2.linear1.weight",
        "ffn2.linear1.bias": f"{p}.feed_forward2.linear1.bias",
        "ffn2.linear2.weight": f"{p}.feed_forward2.linear2.weight",
        "ffn2.linear2.bias": f"{p}.feed_forward2.linear2.bias",
        # ── Final LayerNorm
        "norm.weight": f"{p}.norm_out.weight",
        "norm.bias": f"{p}.norm_out.bias",
    }

    # Map the RPE biases for RelAttention
    mapping.update(
        {
            "mhsa.pos_bias_u": f"{p}.self_attn.pos_bias_u",
            "mhsa.pos_bias_v": f"{p}.self_attn.pos_bias_v",
        }
    )
    return mapping


# STEP 5 — load weights into your block


def load_block_weights(your_block, state_dict, weight_map):
    print("\n" + "═" * 60)
    print("STEP 4 — Loading weights into your ConformerBlock")
    print("═" * 60)

    your_state = your_block.state_dict()
    new_state = {}
    missing = []

    for your_key, nemo_key in weight_map.items():
        if your_key not in your_state:
            print(f"  [SKIP-YOURS]  your key not found : {your_key}")
            continue
        if nemo_key not in state_dict:
            print(f"  [MISSING]     nemo key not found : {nemo_key}")
            missing.append(nemo_key)
            continue

        yt = your_state[your_key]
        nt = state_dict[nemo_key]

        # NeMo stores pointwise convs as Conv1d (out, in, 1) — squeeze to (out, in)
        if nt.shape != yt.shape:
            if nt.dim() == 3 and nt.shape[-1] == 1 and nt.squeeze(-1).shape == yt.shape:
                nt = nt.squeeze(-1)
                print(f"  [SQUEEZE] {your_key:45s} (squeezed kernel dim)")
            else:
                print(f"  [SHAPE MISMATCH] {your_key}")
                print(f"    yours : {yt.shape}   nemo : {nt.shape}")
                missing.append(nemo_key)
                continue

        new_state[your_key] = nt
        print(f"  [OK] {your_key:45s} ← {nemo_key}")

    your_block.load_state_dict(new_state, strict=False)
    return missing


# STEP 6 — forward pass sanity check
def run_forward_test(your_block, d_model):
    print("\n" + "═" * 60)
    print("STEP 5 — Forward pass through your block")
    print("═" * 60)

    your_block.eval()
    torch.manual_seed(42)
    x = torch.randn(1, 50, d_model)

    with torch.no_grad():
        out = your_block(x)

    print(f"  Input  shape : {x.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output mean  : {out.mean().item():.6f}")
    print(f"  Output std   : {out.std().item():.6f}")
    print(f"  First 8 vals : {[round(v, 4) for v in out[0, 0, :8].tolist()]}")

    assert out.shape == x.shape, "Shape mismatch"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print("\n  Shape correct, no NaNs, no Infs.")


# STEP 7 — diff against NeMo (needs nemo_toolkit)


def run_nemo_comparison(nemo_path, block_idx, d_model, n_heads, kernel_size):
    print("\n" + "═" * 60)
    print("STEP 6 — NeMo vs your block (needs nemo_toolkit installed)")
    print("═" * 60)

    if nemo_path.endswith(".ckpt") or nemo_path.endswith(".pt"):
        print("  (Skipping direct NeMo comparison step — requires a full .nemo file)")
        return

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("  nemo_toolkit not installed — skipping direct comparison.")
        print("  pip install nemo_toolkit[asr]  to enable this step.")
        return

    nemo_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo_path)
    nemo_model.eval()
    nemo_block = nemo_model.encoder.layers[block_idx]

    your_block = ConformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        kernel_size=kernel_size,
        dropout=0.0,
        rel_attention=True,  # Use RelAttention for comparison against NeMo
    )
    your_block.load_state_dict(nemo_block.state_dict(), strict=False)
    your_block.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 50, d_model)
    # NeMo's ConformerLayer requires pos_emb and optionally a mask
    # For stt_en_conformer_ctc_small, it's relative positional encoding
    pos_emb = torch.randn(1, 2 * x.size(1) - 1, d_model)
    # NeMo expects a boolean mask (True/1 to mask out, False/0 to keep)
    pad_mask = torch.zeros(1, x.size(1), dtype=torch.bool).to(x.device)

    with torch.no_grad():
        # NeMo layers usually expect (x, mask, pos_emb)
        try:
            nemo_out = nemo_block(x, pad_mask, pos_emb)
        except TypeError:
            # Fallback for different NeMo versions
            nemo_out = nemo_block(x)

        your_out = your_block(x)

    if isinstance(nemo_out, tuple):
        nemo_out = nemo_out[0]

    diff = (nemo_out - your_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  NeMo  first 8: {[round(v, 4) for v in nemo_out[0, 0, :8].tolist()]}")
    print(f"  Yours first 8: {[round(v, 4) for v in your_out[0, 0, :8].tolist()]}")
    print(f"\n  Max  diff : {max_diff:.6f}")
    print(f"  Mean diff : {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("\n  PASS — matches NeMo within tolerance.")
    else:
        print(f"\n  ⚠️  diff = {max_diff:.6f}  (> 1e-4)")
        print("  Likely cause: your attention uses simplified RPE,")
        print("  NeMo uses full Transformer-XL RPE (linear_pos + pos_bias_u/v).")
        print("  FFN and Conv weights should be identical — check those separately.")


# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_path", required=True)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--extract_dir", default="/tmp/nemo_extracted")
    parser.add_argument(
        "--rel_attention", action="store_true", help="Use relative positional attention"
    )
    args = parser.parse_args()

    inspect_nemo(args.nemo_path)
    state_dict = load_nemo_weights(args.nemo_path, args.extract_dir)
    block_keys, prefix = print_key_map(state_dict, args.block_idx)

    if prefix is None:
        print("\n⚠️  Could not detect block prefix. Check STEP 3 output above.")
        return

    d_model, n_heads, kernel_size = 176, 4, 31

    your_block = ConformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        kernel_size=kernel_size,
        dropout=0.0,
        rel_attention=args.rel_attention,
    )

    weight_map = build_weight_map(prefix)
    missing = load_block_weights(your_block, state_dict, weight_map)

    if missing:
        print(f"\n  ⚠️  {len(missing)} keys not loaded — see above.")
    else:
        print("\n  All mapped weights loaded.")

    run_forward_test(your_block, d_model)
    run_nemo_comparison(args.nemo_path, args.block_idx, d_model, n_heads, kernel_size)


if __name__ == "__main__":
    main()
