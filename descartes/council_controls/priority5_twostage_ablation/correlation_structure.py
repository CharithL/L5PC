"""
correlation_structure.py

Phase 5A: Hidden Dimension Correlation Structure

Before running conditional ablation, we need to know which hidden
dimensions are correlated. This module:

1. Computes the full correlation matrix across hidden dimensions
2. Identifies "conditioning sets" for each dimension: other dimensions
   with |r| > 0.3 (moderate or stronger correlation)
3. Flags dimensions that are highly correlated (|r| > 0.7) as
   candidates for confounded ablation results

The conditioning sets are used in Stage 2 (conditional_ablation.py)
to hold correlated dimensions at their conditional distribution
while ablating the target dimension.

Usage:
    python -m descartes.council_controls.priority5_twostage_ablation.correlation_structure
"""

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results/council_controls/phase5_twostage_ablation")
CONDITIONING_THRESHOLD = 0.3   # |r| above this -> include in conditioning set
HIGH_CORR_THRESHOLD = 0.7      # |r| above this -> flag as highly correlated


def compute_correlation_matrix(hidden_states):
    """Compute Pearson correlation matrix across hidden dimensions.

    Args:
        hidden_states: (T, D) numpy array of hidden activations

    Returns:
        (D, D) correlation matrix
    """
    # Handle constant dimensions
    stds = np.std(hidden_states, axis=0)
    valid_mask = stds > 1e-10

    D = hidden_states.shape[1]
    corr = np.eye(D)

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 1:
        sub_corr = np.corrcoef(hidden_states[:, valid_indices].T)
        # Handle NaN from corrcoef
        sub_corr = np.nan_to_num(sub_corr, nan=0.0)
        for i, vi in enumerate(valid_indices):
            for j, vj in enumerate(valid_indices):
                corr[vi, vj] = sub_corr[i, j]

    return corr


def identify_conditioning_sets(corr_matrix, threshold=CONDITIONING_THRESHOLD):
    """For each dimension, identify which other dimensions should be
    conditioned on during ablation.

    Args:
        corr_matrix: (D, D) correlation matrix
        threshold: minimum |r| to include in conditioning set

    Returns:
        dict: {dim_idx: [list of conditioning dim indices]}
    """
    D = corr_matrix.shape[0]
    conditioning_sets = {}

    for i in range(D):
        cond_dims = []
        for j in range(D):
            if i != j and abs(corr_matrix[i, j]) >= threshold:
                cond_dims.append(j)
        conditioning_sets[i] = cond_dims

    return conditioning_sets


def flag_high_correlations(corr_matrix, threshold=HIGH_CORR_THRESHOLD):
    """Flag pairs of dimensions with very high correlation.

    These pairs are most at risk of confounded ablation results:
    ablating one will inevitably affect the other if done unconditionally.

    Returns:
        list of (dim_i, dim_j, r_value) tuples
    """
    D = corr_matrix.shape[0]
    flagged = []
    for i in range(D):
        for j in range(i + 1, D):
            r = corr_matrix[i, j]
            if abs(r) >= threshold:
                flagged.append((int(i), int(j), float(r)))
    return flagged


def analyze_correlation_structure(hidden_states, output_dir=None):
    """Full correlation structure analysis.

    Args:
        hidden_states: (T, D) numpy array of hidden activations

    Returns:
        dict with correlation matrix, conditioning sets, and flags
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    T, D = hidden_states.shape
    print("Analyzing correlation structure: {} timesteps x {} dims".format(T, D))

    # Compute correlation matrix
    corr = compute_correlation_matrix(hidden_states)

    # Conditioning sets
    cond_sets = identify_conditioning_sets(corr)

    # High correlation flags
    high_corr_pairs = flag_high_correlations(corr)

    # Statistics
    upper_triangle = corr[np.triu_indices(D, k=1)]
    abs_upper = np.abs(upper_triangle)

    stats = {
        "n_dims": D,
        "n_timesteps": T,
        "mean_abs_corr": float(np.mean(abs_upper)),
        "median_abs_corr": float(np.median(abs_upper)),
        "max_abs_corr": float(np.max(abs_upper)) if len(abs_upper) > 0 else 0.0,
        "n_pairs_above_03": int(np.sum(abs_upper > 0.3)),
        "n_pairs_above_05": int(np.sum(abs_upper > 0.5)),
        "n_pairs_above_07": int(np.sum(abs_upper > 0.7)),
        "total_pairs": len(upper_triangle),
        "mean_conditioning_set_size": float(np.mean(
            [len(v) for v in cond_sets.values()])),
        "max_conditioning_set_size": int(max(
            len(v) for v in cond_sets.values())) if cond_sets else 0,
        "n_high_corr_flags": len(high_corr_pairs),
    }

    # Save results
    # Correlation matrix as numpy
    np.save(str(out_path / "corr_matrix.npy"), corr)

    # Conditioning sets (convert int keys to str for JSON)
    cond_json = {str(k): v for k, v in cond_sets.items()}
    with open(out_path / "conditioning_sets.json", "w") as f:
        json.dump(cond_json, f, indent=2)

    # High correlation flags
    with open(out_path / "high_corr_flags.json", "w") as f:
        json.dump({
            "threshold": HIGH_CORR_THRESHOLD,
            "pairs": [{"dim_i": p[0], "dim_j": p[1], "r": p[2]}
                      for p in high_corr_pairs],
        }, f, indent=2)

    # Summary stats
    with open(out_path / "correlation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("PHASE 5A: CORRELATION STRUCTURE")
    print("=" * 60)
    print("Dimensions:              {}".format(D))
    print("Mean |r|:                {:.3f}".format(stats["mean_abs_corr"]))
    print("Pairs |r| > 0.3:        {} / {}".format(
        stats["n_pairs_above_03"], stats["total_pairs"]))
    print("Pairs |r| > 0.7:        {}".format(stats["n_pairs_above_07"]))
    print("Mean conditioning size:  {:.1f}".format(
        stats["mean_conditioning_set_size"]))
    print("Max conditioning size:   {}".format(
        stats["max_conditioning_set_size"]))

    if high_corr_pairs:
        print("\nHighly correlated pairs (|r| > {}):"
              .format(HIGH_CORR_THRESHOLD))
        for dim_i, dim_j, r in high_corr_pairs[:20]:
            print("  dims ({}, {}): r = {:.3f}".format(dim_i, dim_j, r))
        if len(high_corr_pairs) > 20:
            print("  ... and {} more".format(len(high_corr_pairs) - 20))

    print("\nResults saved to: {}".format(out_path))
    return {"corr_matrix": corr, "conditioning_sets": cond_sets,
            "high_corr_pairs": high_corr_pairs, "stats": stats}


def load_conditioning_sets(path=None):
    """Load conditioning sets from saved JSON for use by conditional_ablation."""
    if path is None:
        path = RESULTS_DIR / "conditioning_sets.json"
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Standalone demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic hidden states for demo...")
    rng = np.random.RandomState(42)
    D = 128
    T = 2000

    # Create hidden states with known correlation structure
    # Some dimensions are correlated, others are independent
    base = rng.randn(T, D)
    # Inject correlations: dims 0-10 correlated, 50-60 correlated
    for i in range(1, 10):
        base[:, i] = 0.6 * base[:, 0] + 0.4 * rng.randn(T)
    for i in range(51, 60):
        base[:, i] = 0.5 * base[:, 50] + 0.5 * rng.randn(T)

    analyze_correlation_structure(base)
