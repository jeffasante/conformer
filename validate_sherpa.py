import sys
import argparse
import os
import onnx
import torch
import torch.nn as nn
import numpy as np
from onnx import numpy_helper

from conformer_block import ConformerBlock


def load_onnx_as_state_dict(onnx_path: str):
    """Convert ONNX initializers to a PyTorch-like state dict."""
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    state_dict = {}
    for initializer in model.graph.initializer:
        # Initializers are numpy arrays
        tensor = numpy_helper.to_array(initializer)
        state_dict[initializer.name] = torch.from_numpy(tensor).clone()
    return state_dict


def get_block_keys(state_dict, block_idx=0):
    prefix = f"encoder.encoder.layers.{block_idx}"
    block_keys = {k: v.shape for k, v in state_dict.items() if k.startswith(prefix)}
    return block_keys, prefix


def build_sherpa_map(prefix, rel_attention=False):
    p = prefix
    mapping = {
        # FFN Macaron (FFN1)
        "ffn1.linear1.weight": f"{p}.feed_forward_macaron.0.weight",
        "ffn1.linear1.bias":   f"{p}.feed_forward_macaron.0.bias",
        "ffn1.linear2.weight": f"{p}.feed_forward_macaron.4.weight",
        "ffn1.linear2.bias":   f"{p}.feed_forward_macaron.4.bias",
        "ffn1.norm.weight":    f"{p}.norm_ff_macaron.weight",
        "ffn1.norm.bias":      f"{p}.norm_ff_macaron.bias",

        # MHSA Norm (Shared name)
        "mhsa.norm.weight":        f"{p}.norm_self_att.weight",
        "mhsa.norm.bias":          f"{p}.norm_self_att.bias",

        # Conv
        "conv.norm.weight"         : f"{p}.norm_conv.weight",
        "conv.norm.bias"           : f"{p}.norm_conv.bias",
        "conv.pw_conv1.weight"     : f"{p}.conv_module.pointwise_conv1.weight",
        "conv.pw_conv1.bias"       : f"{p}.conv_module.pointwise_conv1.bias",
        "conv.dw_conv.weight"      : f"{p}.conv_module.depthwise_conv.weight",
        "conv.dw_conv.bias"        : f"{p}.conv_module.depthwise_conv.bias",
        "conv.bn.weight"           : f"{p}.conv_module.batch_norm.weight",
        "conv.bn.bias"             : f"{p}.conv_module.batch_norm.bias",
        "conv.bn.running_mean"     : f"{p}.conv_module.batch_norm.running_mean",
        "conv.bn.running_var"      : f"{p}.conv_module.batch_norm.running_var",
        "conv.pw_conv2.weight"     : f"{p}.conv_module.pointwise_conv2.weight",
        "conv.pw_conv2.bias"       : f"{p}.conv_module.pointwise_conv2.bias",

        # FFN (FFN2)
        "ffn2.linear1.weight": f"{p}.feed_forward.0.weight",
        "ffn2.linear1.bias":   f"{p}.feed_forward.0.bias",
        "ffn2.linear2.weight": f"{p}.feed_forward.4.weight",
        "ffn2.linear2.bias":   f"{p}.feed_forward.4.bias",
        "ffn2.norm.weight":    f"{p}.norm_ff.weight",
        "ffn2.norm.bias":      f"{p}.norm_ff.bias",

        # Final Norm
        "norm.weight": f"{p}.norm_final.weight",
        "norm.bias":   f"{p}.norm_final.bias",
    }
    
    # MHSA Projection naming differences
    if rel_attention:
        mapping.update({
            "mhsa.linear_out.weight" : f"{p}.self_attn.out_proj.weight",
            "mhsa.linear_out.bias"   : f"{p}.self_attn.out_proj.bias",
            "mhsa.pos_bias_u": f"{p}.self_attn.pos_bias_u",
            "mhsa.pos_bias_v": f"{p}.self_attn.pos_bias_v",
            "mhsa.linear_pos.bias"   : f"{p}.self_attn.linear_pos.bias",
        })
    else:
        mapping.update({
            "mhsa.out_proj.weight":    f"{p}.self_attn.out_proj.weight",
            "mhsa.out_proj.bias":      f"{p}.self_attn.out_proj.bias",
        })
        
    return mapping


def load_and_validate(onnx_path, rel_attention=False):
    state_dict = load_onnx_as_state_dict(onnx_path)
    block_keys, prefix = get_block_keys(state_dict, 0)

    if not block_keys:
        print("Could not find block 0 keys. Printing keys to help debugging:")
        for k in list(state_dict.keys())[:50]: print(f"  {k}")
        return

    # Sherpa small config
    d_model = 384
    n_heads = 8
    kernel_size = 31

    your_block = ConformerBlock(
        d_model=d_model, 
        n_heads=n_heads, 
        kernel_size=kernel_size, 
        dropout=0.0,
        rel_attention=rel_attention
    )
    your_state = your_block.state_dict()

    mapping = build_sherpa_map(prefix, rel_attention)
    new_state = {}

    for py_k, onnx_k in mapping.items():
        if onnx_k in state_dict:
            onnx_t = state_dict[onnx_k]
            # Handle pointwise: if ONNX is 2D but PyTorch expects 3D (Conv1d k=1)
            if (
                onnx_t.dim() == 2
                and your_state[py_k].dim() == 3
                and your_state[py_k].shape[-1] == 1
            ):
                onnx_t = onnx_t.unsqueeze(-1)
                print(f"  [UNSQUEEZE] {py_k:30s} (added kernel dim)")

            if py_k in your_state and your_state[py_k].shape == onnx_t.shape:
                new_state[py_k] = onnx_t
                print(f"  [OK] {py_k:30s} loaded")
            else:
                print(
                    f"  [SHAPE MISMATCH] {py_k} (ours: {your_state.get(py_k, 'N/A').shape}, onnx: {onnx_t.shape})"
                )

    # Handle Fused QKV (in_proj)
    in_proj_w = state_dict.get(f"{prefix}.self_attn.in_proj.weight")
    in_proj_b = state_dict.get(f"{prefix}.self_attn.in_proj.bias")
    if in_proj_w is not None and in_proj_b is not None:
        # Split into 3 parts for Q, K, V
        q_w, k_w, v_w = torch.chunk(in_proj_w, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(in_proj_b, 3, dim=0)
        
        if rel_attention:
            new_state["mhsa.linear_q.weight"] = q_w
            new_state["mhsa.linear_q.bias"]   = q_b
            new_state["mhsa.linear_k.weight"] = k_w
            new_state["mhsa.linear_k.bias"]   = k_b
            new_state["mhsa.linear_v.weight"] = v_w
            new_state["mhsa.linear_v.bias"]   = v_b
            print("  [OK] mhsa.linear_q/k/v loaded via in_proj split")
        else:
            new_state["mhsa.q_proj.weight"] = q_w
            new_state["mhsa.q_proj.bias"]   = q_b
            new_state["mhsa.k_proj.weight"] = k_w
            new_state["mhsa.k_proj.bias"]   = k_b
            new_state["mhsa.v_proj.weight"] = v_w
            new_state["mhsa.v_proj.bias"]   = v_b
            print("  [OK] mhsa.q/k/v_proj loaded via in_proj split")

    your_block.load_state_dict(new_state, strict=False)

    # Validation Pass
    your_block.eval()
    x = torch.randn(1, 50, d_model)
    with torch.no_grad():
        out = your_block(x)

    print("\n--- Validation Results ---")
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Output mean  : {out.mean().item():.6f}")
    if not torch.isnan(out).any():
        print("BLOCK RUNS SUCCESSFULLY")
    else:
        print("OUTPUT CONTAINS NaNs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="sherpa-onnx-streaming-conformer-zh-2023-05-23/encoder-epoch-99-avg-1.onnx")
    parser.add_argument("--rel_attention", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.onnx_path):
        load_and_validate(args.onnx_path, args.rel_attention)
    else:
        print(f"Could not find model at {args.onnx_path}")
