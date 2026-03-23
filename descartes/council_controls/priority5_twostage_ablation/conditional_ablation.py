"""
conditional_ablation.py

Phase 5B: Stage 2 Conditional Ablation

Standard (Stage 1) ablation resamples a target dimension from its marginal
distribution. But if the target is correlated with other dimensions, this
breaks the joint distribution and can produce spurious effects.

Stage 2 ablation resamples the target dimension WHILE HOLDING conditioning
dimensions at their conditional distribution, using binning/quantile
stratification:

  1. Discretize each conditioning dimension into quantile bins
  2. For each combination of conditioning bin values, collect the
     distribution of the target dimension within that stratum
  3. Resample the target from the WITHIN-STRATUM distribution

If a variable passes Stage 1 (unconditional) but fails Stage 2
(conditional), it is classified as INDIRECT: its apparent importance
was mediated through correlated dimensions.

Usage:
    python -m descartes.council_controls.priority5_twostage_ablation.conditional_ablation
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .correlation_structure import load_conditioning_sets

RESULTS_DIR = Path("results/council_controls/phase5_twostage_ablation")
N_QUANTILE_BINS = 5     # Number of bins per conditioning dimension
MAX_COND_DIMS = 3       # Max conditioning dims to use (avoid curse of dimensionality)
RIDGE_ALPHA = 1.0
N_CV_FOLDS = 5
N_ABLATION_REPS = 10    # Repetitions of ablation for stability


def discretize_dimension(values, n_bins=N_QUANTILE_BINS):
    """Discretize a continuous dimension into quantile bins.

    Returns:
        bin_labels: integer array of bin assignments (0 to n_bins-1)
        bin_edges: array of bin edge values
    """
    # Use quantiles for equal-frequency bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, percentiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return np.zeros(len(values), dtype=int), bin_edges
    bin_labels = np.digitize(values, bin_edges[1:-1])
    return bin_labels, bin_edges


def conditional_resample(hidden_states, target_dim, conditioning_dims,
                         n_bins=N_QUANTILE_BINS, rng=None):
    """Resample target dimension conditional on binned conditioning dims.

    For each unique combination of conditioning bin values, replace the
    target dimension with a random draw from the target's distribution
    within that stratum.

    Args:
        hidden_states: (T, D) array
        target_dim: int, dimension to ablate
        conditioning_dims: list of ints, dimensions to condition on
        n_bins: quantile bins per conditioning dimension
        rng: numpy RandomState

    Returns:
        ablated_states: (T, D) array with target conditionally resampled
    """
    if rng is None:
        rng = np.random.RandomState(0)

    T, D = hidden_states.shape
    ablated = hidden_states.copy()

    if not conditioning_dims:
        # No conditioning — fall back to marginal resample
        ablated[:, target_dim] = rng.permutation(
            hidden_states[:, target_dim])
        return ablated

    # Limit conditioning dims to avoid empty strata
    cond_dims = conditioning_dims[:MAX_COND_DIMS]

    # Discretize conditioning dimensions
    bin_arrays = []
    for cd in cond_dims:
        bins, _ = discretize_dimension(hidden_states[:, cd], n_bins)
        bin_arrays.append(bins)

    # Create composite stratum key per timestep
    strata = np.zeros(T, dtype=int)
    for i, bins in enumerate(bin_arrays):
        strata = strata * (n_bins + 1) + bins

    unique_strata = np.unique(strata)

    for s in unique_strata:
        mask = strata == s
        n_in_stratum = np.sum(mask)
        if n_in_stratum < 2:
            continue
        # Resample target from within-stratum distribution
        stratum_values = hidden_states[mask, target_dim]
        ablated[mask, target_dim] = rng.choice(
            stratum_values, size=n_in_stratum, replace=True)

    return ablated


def probe_r2(hidden_states, probe_target, alpha=RIDGE_ALPHA,
             n_folds=N_CV_FOLDS):
    """Compute cross-validated Ridge R2 for probe target."""
    T_h = hidden_states.shape[0]
    T_y = len(probe_target)
    min_len = min(T_h, T_y)
    H = hidden_states[:min_len]
    y = probe_target[:min_len]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(H):
        ridge = Ridge(alpha=alpha)
        ridge.fit(H[train_idx], y[train_idx])
        scores.append(ridge.score(H[test_idx], y[test_idx]))
    return float(np.mean(scores))


def stage2_ablation(hidden_states, probe_target, target_dim,
                    conditioning_sets, n_reps=N_ABLATION_REPS):
    """Run Stage 2 (conditional) ablation for a single target dimension.

    Args:
        hidden_states: (T, D) array
        probe_target: (T,) array — the variable being probed
        target_dim: int — hidden dimension to ablate
        conditioning_sets: dict {dim: [cond_dims]}
        n_reps: number of ablation repetitions

    Returns:
        dict with baseline R2, ablated R2 stats, delta
    """
    baseline_r2 = probe_r2(hidden_states, probe_target)

    cond_dims = conditioning_sets.get(target_dim, [])

    ablated_r2s = []
    for rep in range(n_reps):
        rng = np.random.RandomState(rep)
        ablated = conditional_resample(
            hidden_states, target_dim, cond_dims, rng=rng)
        r2 = probe_r2(ablated, probe_target)
        ablated_r2s.append(r2)

    mean_ablated = float(np.mean(ablated_r2s))
    delta = baseline_r2 - mean_ablated

    return {
        "target_dim": target_dim,
        "n_conditioning_dims": len(cond_dims[:MAX_COND_DIMS]),
        "conditioning_dims": cond_dims[:MAX_COND_DIMS],
        "baseline_r2": baseline_r2,
        "ablated_r2_mean": mean_ablated,
        "ablated_r2_std": float(np.std(ablated_r2s, ddof=1)) if len(ablated_r2s) > 1 else 0.0,
        "delta_r2": delta,
        "n_reps": n_reps,
    }


def run_conditional_ablation(hidden_states, probe_target,
                             dims_to_test=None, conditioning_sets=None,
                             output_dir=None):
    """Run Stage 2 conditional ablation across multiple dimensions.

    Args:
        hidden_states: (T, D) array
        probe_target: (T,) probe target
        dims_to_test: list of dim indices (default: all)
        conditioning_sets: pre-computed from correlation_structure

    Returns:
        list of per-dimension results
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    D = hidden_states.shape[1]
    if dims_to_test is None:
        dims_to_test = list(range(D))
    if conditioning_sets is None:
        conditioning_sets = load_conditioning_sets()

    results = []
    for i, dim in enumerate(dims_to_test):
        if (i + 1) % 20 == 0 or i == 0:
            print("  Stage 2 ablation: dim {}/{} ({})".format(
                i + 1, len(dims_to_test), dim))
        r = stage2_ablation(hidden_states, probe_target, dim,
                            conditioning_sets)
        results.append(r)

    out_file = out_path / "conditional_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nStage 2 results saved: {}".format(out_file))
    return results


if __name__ == "__main__":
    print("Running Stage 2 conditional ablation demo...")
    rng = np.random.RandomState(42)
    D, T = 32, 1000

    # Synthetic hidden states with correlation
    H = rng.randn(T, D).astype(np.float32)
    for i in range(1, 5):
        H[:, i] = 0.6 * H[:, 0] + 0.4 * rng.randn(T)

    # Probe target depends on dim 0 directly
    y = 0.8 * H[:, 0] + 0.2 * rng.randn(T)

    cond_sets = {i: [] for i in range(D)}
    cond_sets[0] = [1, 2, 3, 4]
    for i in range(1, 5):
        cond_sets[i] = [0]

    results = run_conditional_ablation(
        H, y, dims_to_test=list(range(8)), conditioning_sets=cond_sets)

    for r in results:
        print("  dim {}: baseline={:.3f}, ablated={:.3f}, delta={:.3f}".format(
            r["target_dim"], r["baseline_r2"],
            r["ablated_r2_mean"], r["delta_r2"]))
