#!/usr/bin/env python
"""
Extract and analyze attention residual weights from Post AR and Full AR TSP100 models.

Usage:
  python scripts/analyze_attention_weights.py \
      --model_type post_ar \
      --checkpoint <path> \
      --test_data <path> \
      --output_dir <path> \
      --batch_size 128 --device cuda --save_raw --plot --verify
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parents[1] / 'code/POMO_PO'
_pomo_tsp = _project_root / "POMO" / "TSP" / "POMO"
sys.path.insert(0, str(_pomo_tsp))
sys.path.insert(0, str(_project_root / "POMO" / "TSP"))
sys.path.insert(0, str(_project_root / "POMO"))

from TSPEnv import TSPEnv


# ═══════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "post_ar": {
        "name": "Post AR",
        "model_module": "TSPModelPostAR",
        "num_layers": 6,
        "num_sources": 7,  # H0 + H1..H6
        "source_names": ["Input", "Layer 1", "Layer 2", "Layer 3",
                         "Layer 4", "Layer 5", "Layer 6"],
        "full_ar_grouped_names": None,
    },
    "full_ar": {
        "name": "Full AR",
        "model_module": "TSPModelFullAR",
        "num_layers": 6,
        "num_sources": 13,  # H0 + 6*(MHA,FFN)
        "source_names": None,  # Filled dynamically
        "full_ar_grouped_names": ["Input", "Layer 1", "Layer 2", "Layer 3",
                                  "Layer 4", "Layer 5", "Layer 6"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_commit(repo_root):
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ).decode().strip()
    except Exception:
        return "unknown"


def load_model_class(module_name):
    import importlib
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "TSPModel"):
        raise AttributeError(f"Module '{module_name}' must define TSPModel")
    return getattr(mod, "TSPModel")


def build_model(model_type, device):
    """Build model with record_attention enabled."""
    cfg = MODEL_CONFIGS[model_type]
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": cfg["num_layers"],
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 50,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
        "record_attention": True,       # <── ENABLE attention recording
    }

    ModelCls = load_model_class(cfg["model_module"])
    model = ModelCls(**model_params).to(device)
    return model


def build_model_no_record(model_type, device):
    """Build identical model WITHOUT attention recording for verification."""
    cfg = MODEL_CONFIGS[model_type]
    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** 0.5,
        "encoder_layer_num": cfg["num_layers"],
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 50,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
        "record_attention": False,
    }

    ModelCls = load_model_class(cfg["model_module"])
    model = ModelCls(**model_params).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint; return (model, loaded_epoch)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch", 0)
    return epoch


def run_inference(model, env, batch_size, device, num_instances=None):
    """Run inference and collect recorded attention weights.

    Returns:
        all_weights: list of numpy arrays, each [batch_size, N, num_sources]
        all_paths: list of path tensors
        all_lengths: list of path length scalars
    """
    all_weights = []
    all_paths = []
    all_lengths = []

    total = num_instances if num_instances else len(env.loaded_problems)
    env.batch_count = 0
    processed = 0

    while processed < total:
        remaining = total - processed
        b = min(batch_size, remaining)

        env.load_problems(b)
        problems = env.problems.clone()

        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        # ── Collect encoder attention weights ──
        weights = model.encoder._recorded_weights.clone()  # [B, N, K]
        all_weights.append(weights.numpy())

        # ── Run full decoding to capture paths ──
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

        max_reward, max_idx = reward.max(dim=1)
        batch_paths = env.selected_node_list[
            torch.arange(b, device=device), max_idx
        ]  # [B, problem]
        batch_lengths = -max_reward.float()

        all_paths.append(batch_paths.cpu())
        all_lengths.append(batch_lengths.cpu())

        processed += b

        # Clean up stored weights to free GPU mem
        del model.encoder._recorded_weights
        model.encoder._recorded_weights = None

    return all_weights, all_paths, all_lengths


def run_inference_no_record(model, env, batch_size, device, num_instances=None):
    """Run inference WITHOUT attention recording (for consistency check)."""
    all_paths = []
    all_lengths = []
    total = num_instances if num_instances else len(env.loaded_problems)
    env.batch_count = 0
    processed = 0

    while processed < total:
        remaining = total - processed
        b = min(batch_size, remaining)

        env.load_problems(b)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)

        max_reward, max_idx = reward.max(dim=1)
        batch_paths = env.selected_node_list[
            torch.arange(b, device=device), max_idx
        ]
        batch_lengths = -max_reward.float()

        all_paths.append(batch_paths.cpu())
        all_lengths.append(batch_lengths.cpu())
        processed += b

    return all_paths, all_lengths


# ═══════════════════════════════════════════════════════════════════════
# Full AR source order
# ═══════════════════════════════════════════════════════════════════════

def get_full_ar_source_order():
    """Return the true source order for Full AR model (13 sources).

    Verified from TSPModelFullAR.TSP_Encoder.forward():
      Source 0:  H0 (init_state)
      Source 1:  MHA_out L1
      Source 2:  FFN_out L1
      Source 3:  MHA_out L2
      Source 4:  FFN_out L2
      Source 5:  MHA_out L3
      Source 6:  FFN_out L3
      Source 7:  MHA_out L4
      Source 8:  FFN_out L4
      Source 9:  MHA_out L5
      Source 10: FFN_out L5
      Source 11: MHA_out L6
      Source 12: FFN_out L6  (LAST FF FUSION POINT)
    """
    names = [
        "H0",
        "MHA(L1)", "FFN(L1)",
        "MHA(L2)", "FFN(L2)",
        "MHA(L3)", "FFN(L3)",
        "MHA(L4)", "FFN(L4)",
        "MHA(L5)", "FFN(L5)",
        "MHA(L6)", "FFN(L6)",
    ]
    return names


# ═══════════════════════════════════════════════════════════════════════
# Full AR → 7-group aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate_full_ar_to_7(raw_weights):
    """Aggregate Full AR 13-source weights into 7 layer-level groups.

    Mapping (verified from source order):
      group 0 = w[H0]                          (index 0)
      group 1 = w[MHA(L1)] + w[FFN(L1)]       (index 1+2)
      group 2 = w[MHA(L2)] + w[FFN(L2)]       (index 3+4)
      group 3 = w[MHA(L3)] + w[FFN(L3)]       (index 5+6)
      group 4 = w[MHA(L4)] + w[FFN(L4)]       (index 7+8)
      group 5 = w[MHA(L5)] + w[FFN(L5)]       (index 9+10)
      group 6 = w[MHA(L6)] + w[FFN(L6)]       (index 11+12)

    Args:
        raw_weights: [N_instances, N_nodes, 13]

    Returns:
        grouped: [N_instances, N_nodes, 7]
    """
    groups = [
        [0],
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
    ]
    grouped = np.stack(
        [raw_weights[..., g].sum(axis=-1) for g in groups],
        axis=-1,
    )
    return grouped


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

def compute_stats(weights_array, source_names):
    """
    Args:
        weights_array: [N_instances, N_nodes, K]
        source_names: list of K names

    Returns:
        dict with keys: global_mean, per_instance_mean, std_over_instances,
                        std_over_all_nodes, max_abs_error
    """
    N_inst, N_nodes, K = weights_array.shape

    # Weight sum check
    weight_sums = weights_array.sum(axis=-1)  # [N_inst, N_nodes]
    max_abs_error = np.max(np.abs(weight_sums - 1.0))

    # Global mean (over all instances and nodes)
    global_mean = weights_array.mean(axis=(0, 1))  # [K]

    # Per-instance mean (each instance averaged over 100 nodes)
    per_instance_mean = weights_array.mean(axis=1)  # [N_inst, K]

    # Std over instances (of per-instance means)
    std_over_instances = per_instance_mean.std(axis=0)  # [K]

    # Std over all nodes (all instances × nodes pooled)
    std_over_all_nodes = weights_array.reshape(-1, K).std(axis=0)  # [K]

    return {
        "global_mean": global_mean,
        "per_instance_mean": per_instance_mean,
        "std_over_instances": std_over_instances,
        "std_over_all_nodes": std_over_all_nodes,
        "max_abs_error": max_abs_error,
        "source_names": source_names,
    }


def save_summary_csv(stats, filepath):
    """Save summary CSV."""
    names = stats["source_names"]
    rows = [
        "source,mean_weight,std_over_instances,std_over_all_nodes",
    ]
    for k, name in enumerate(names):
        rows.append(
            f"{name},{stats['global_mean'][k]:.6f},"
            f"{stats['std_over_instances'][k]:.6f},"
            f"{stats['std_over_all_nodes'][k]:.6f}"
        )
    with open(filepath, "w") as f:
        f.write("\n".join(rows) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract TSP100 attention residual weights"
    )
    parser.add_argument("--model_type", required=True, choices=["post_ar", "full_ar"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--test_data", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_instances", type=int, default=None,
                        help="Number of instances to process (default: all)")
    parser.add_argument("--save_raw", action="store_true",
                        help="Save per-instance per-node raw weights")
    parser.add_argument("--plot", action="store_true",
                        help="Generate attention weight figure")
    parser.add_argument("--verify", action="store_true",
                        help="Verify inference consistency with/without recording")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_type]
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    # ── Output dir ──
    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.overwrite:
        print(f"ERROR: Output directory {out_dir} already exists. Use --overwrite.")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Attention Weight Analysis: {cfg['name']}")
    print(f"{'='*60}")
    print(f"  Model type:    {args.model_type}")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Test data:     {args.test_data}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Device:        {device}")
    print(f"  Num sources:   {cfg['num_sources']}")

    # ── Verify checkpoint and test data exist ──
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.test_data):
        print(f"ERROR: Test data not found: {args.test_data}")
        sys.exit(1)

    # Select the recorded historical definition before constructing the model.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    if args.model_type == "full_ar" and "encoder.final_norm.weight" not in state:
        cfg["model_module"] = "TSPModelFullARLegacy"

    # ── Build model ──
    print(f"\n  Building model ({cfg['model_module']})...")
    model = build_model(args.model_type, device)
    epoch = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"  Loaded checkpoint (epoch {epoch}), {cfg['num_layers']} layers")

    # ── Verification: inference consistency check ──
    if args.verify:
        print(f"\n  ── Verification Mode ──")
        print(f"  Running on 32 instances with record ON vs OFF...")

        env_rec = TSPEnv(
            problem_size=100, pomo_size=100, device=device,
            validation_set_path=args.test_data,
        )
        env_no_rec = TSPEnv(
            problem_size=100, pomo_size=100, device=device,
            validation_set_path=args.test_data,
        )

        w_rec, p_rec, l_rec = run_inference(
            model, env_rec, 32, device, num_instances=32
        )

        model_no_rec = build_model_no_record(args.model_type, device)
        load_checkpoint(model_no_rec, args.checkpoint, device)
        model_no_rec.eval()

        p_no_rec, l_no_rec = run_inference_no_record(
            model_no_rec, env_no_rec, 32, device, num_instances=32
        )

        # Compare
        paths_match = all(
            torch.equal(p1, p2) for p1, p2 in zip(p_rec, p_no_rec)
        )
        lengths_close = all(
            torch.allclose(l1, l2, rtol=1e-5)
            for l1, l2 in zip(l_rec, l_no_rec)
        )

        print(f"  Paths match:       {paths_match}")
        print(f"  Lengths match:     {lengths_close}")

        # Weight shape and sum check
        w_all = np.concatenate(w_rec, axis=0)
        print(f"  Weight shape:      {w_all.shape}")
        weight_sum = w_all.sum(axis=-1)
        max_err = np.max(np.abs(weight_sum - 1.0))
        print(f"  Max |sum(w) - 1|:  {max_err:.8f}")

        del model_no_rec, env_no_rec

        if not (paths_match and lengths_close):
            print("ERROR: Inference inconsistency detected! Aborting.")
            sys.exit(1)
        print("  Verification PASSED.\n")

    # ── Full run ──
    print(f"\n  ── Running full analysis ──")
    env = TSPEnv(
        problem_size=100, pomo_size=100, device=device,
        validation_set_path=args.test_data,
    )

    total_instances = args.num_instances or len(env.loaded_problems)
    print(f"  Instances: {total_instances}")

    t_start = time.time()

    weights_list, paths_list, lengths_list = run_inference(
        model, env, args.batch_size, device, num_instances=total_instances
    )

    t_elapsed = time.time() - t_start
    print(f"  Finished in {t_elapsed:.1f}s")

    # ── Concatenate weights ──
    all_w = np.concatenate(weights_list, axis=0)  # [N_inst, N_nodes, K]

    # Trim to actual count if needed
    all_w = all_w[:total_instances]

    print(f"  Raw weight shape: {all_w.shape}")
    weight_sum = all_w.sum(axis=-1)
    max_err = np.max(np.abs(weight_sum - 1.0))
    print(f"  Max |sum(w) - 1|: {max_err:.8f}")

    # ── Save config ──
    config = {
        "model_type": args.model_type,
        "model_name": cfg["name"],
        "checkpoint_path": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "test_data_path": args.test_data,
        "test_data_sha256": file_sha256(args.test_data),
        "num_layers": cfg["num_layers"],
        "num_sources": cfg["num_sources"],
        "source_names": cfg["source_names"],
        "batch_size": args.batch_size,
        "num_instances": total_instances,
        "device": str(device),
        "inference_time_s": t_elapsed,
        "max_abs_weight_error": float(max_err),
        "git_commit": get_git_commit(_project_root),
    }

    if args.model_type == "full_ar":
        source_order = get_full_ar_source_order()
        config["full_ar_source_order"] = source_order
        with open(out_dir / "full_ar_source_order.json", "w") as f:
            json.dump({"source_order": source_order}, f, indent=2)

    with open(out_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Save raw weights ──
    if args.save_raw:
        raw_path = out_dir / "post_ar_raw_weights.npy" if args.model_type == "post_ar" else out_dir / "full_ar_raw_final_ff_weights.npy"
        np.save(raw_path, all_w)
        print(f"  Saved raw weights: {raw_path}")

    # ── Statistics ──
    if args.model_type == "post_ar":
        source_names = cfg["source_names"]
        stats = compute_stats(all_w, source_names)

        per_inst_mean = stats["per_instance_mean"]
        np.save(out_dir / "post_ar_per_instance_mean.npy", per_inst_mean)
        save_summary_csv(stats, out_dir / "post_ar_summary.csv")

        print(f"\n  ── Post AR Summary ──")
        print(f"  {'Source':<12s} {'Mean':>8s}  {'Std(inst)':>10s}  {'Std(all)':>10s}")
        for k, name in enumerate(source_names):
            print(f"  {name:<12s} {stats['global_mean'][k]:8.4f}  "
                  f"{stats['std_over_instances'][k]:10.4f}  "
                  f"{stats['std_over_all_nodes'][k]:10.4f}")

        print(f"\n  Max |sum(w) - 1| = {max_err:.8f}")

        # Store for plotting
        post_ar_stats_for_plot = stats

    elif args.model_type == "full_ar":
        # Group into 7 layer-level groups
        grouped_w = aggregate_full_ar_to_7(all_w)  # [N_inst, N_nodes, 7]
        print(f"  Grouped weight shape: {grouped_w.shape}")

        # Check group sum
        group_sum = grouped_w.sum(axis=-1)
        group_max_err = np.max(np.abs(group_sum - 1.0))
        print(f"  Grouped max |sum(w) - 1|: {group_max_err:.8f}")

        grouped_source_names = ["Input", "Layer 1", "Layer 2", "Layer 3",
                                "Layer 4", "Layer 5", "Layer 6"]
        grouped_stats = compute_stats(grouped_w, grouped_source_names)

        np.save(out_dir / "full_ar_grouped_weights.npy", grouped_w)
        np.save(out_dir / "full_ar_per_instance_grouped_mean.npy",
                grouped_stats["per_instance_mean"])
        save_summary_csv(grouped_stats, out_dir / "full_ar_grouped_summary.csv")

        print(f"\n  ── Full AR Grouped Summary ──")
        print(f"  {'Source':<12s} {'Mean':>8s}  {'Std(inst)':>10s}  {'Std(all)':>10s}")
        for k, name in enumerate(grouped_source_names):
            print(f"  {name:<12s} {grouped_stats['global_mean'][k]:8.4f}  "
                  f"{grouped_stats['std_over_instances'][k]:10.4f}  "
                  f"{grouped_stats['std_over_all_nodes'][k]:10.4f}")

        print(f"\n  Raw max |sum(w) - 1| = {max_err:.8f}")
        print(f"  Grouped max |sum(w) - 1| = {group_max_err:.8f}")

        # Store for plotting (only valid if post_ar was also run separately;
        # we still compute stats for this model)
        full_ar_grouped_stats_for_plot = grouped_stats

        stats = compute_stats(all_w, get_full_ar_source_order())  # 13-source stats

    # ── Plot: delegate to beautified plot module ──
    if args.plot:
        _scripts_dir = str(Path(__file__).resolve().parent)
        sys.path.insert(0, _scripts_dir)
        from plot_combined_attention import plot_single
        plot_single(str(out_dir), cfg["name"], str(out_dir), dpi=300)

    # ── Final report ──
    print(f"\n{'='*60}")
    print(f"  Analysis complete")
    print(f"  Output: {out_dir}")
    print(f"  Files written:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
