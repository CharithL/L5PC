"""
twostage_classify.py

Phase 5C: Two-Stage Classification (Unconditional + Conditional Ablation)

Combines Stage 1 (unconditional marginal ablation) and Stage 2 (conditional
ablation from conditional_ablation.py) to produce a refined classification
of hidden dimensions:

  MANDATORY  — Passes both Stage 1 and Stage 2. Ablation causes R2 drop
               even after controlling for correlated dimensions.
               Strong evidence the dimension encodes the target directly.

  ZOMBIE     — Fails Stage 1. No detectable effect even with marginal ablation.
               Dimension does not contribute to probe decoding.

  INDIRECT   — Passes Stage 1, FAILS Stage 2. The dimension appears
               important when ablated unconditionally, but the effect
               disappears when conditioning on correlated dimensions.
               The dimension's apparent importance was mediated through
               shared variance with other dimensions.

The INDIRECT category is the key contribution of Phase 5: it catches
dimensions that would be falsely classified as mandatory by naive ablation.

Usage:
    python -m descartes.council_controls.priority5_twostage_ablation.twostage_classify
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .correlation_structure import (
    analyze_correlation_structure, load_conditioning_sets,
)
from .conditional_ablation import (
    probe_r2, conditional_resample, N_ABLATION_REPS, MAX_COND_DIMS,
)

RESULTS_DIR = Path("results/council_controls/phase5_twostage_ablation")

# Classification thresholds
STAGE1_DELTA_THRESHOLD = 0.02   # Minimum R2 drop for Stage 1 significance
STAGE2_DELTA_THRESHOLD = 0.02   # Minimum R2 drop for Stage 2 significance
RIDGE_ALPHA = 1.0
N_CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Stage 1: Unconditional (marginal) ablation
# ---------------------------------------------------------------------------

def stage1_unconditional(hidden_states, probe_target, target_dim,
                          n_reps=N_ABLATION_REPS):
    """Stage 1: ablate target dim by random permutation (marginal).

    This is standard ablation — no conditioning on correlated dims.
    """
    baseline_r2 = probe_r2(hidden_states, probe_target)

    ablated_r2s = []
    for rep in range(n_reps):
        rng = np.random.RandomState(rep + 1000)
        H_abl = hidden_states.copy()
        H_abl[:, target_dim] = rng.permutation(H_abl[:, target_dim])
        r2 = probe_r2(H_abl, probe_target)
        ablated_r2s.append(r2)

    mean_ablated = float(np.mean(ablated_r2s))
    delta = baseline_r2 - mean_ablated

    return {
        "baseline_r2": baseline_r2,
        "stage1_ablated_r2": mean_ablated,
        "stage1_delta": delta,
        "stage1_passes": delta >= STAGE1_DELTA_THRESHOLD,
    }


def stage2_conditional(hidden_states, probe_target, target_dim,
                        conditioning_sets, n_reps=N_ABLATION_REPS):
    """Stage 2: ablate target dim conditional on correlated dims."""
    baseline_r2 = probe_r2(hidden_states, probe_target)
    cond_dims = conditioning_sets.get(target_dim, [])

    ablated_r2s = []
    for rep in range(n_reps):
        rng = np.random.RandomState(rep + 2000)
        H_abl = conditional_resample(
            hidden_states, target_dim, cond_dims, rng=rng)
        r2 = probe_r2(H_abl, probe_target)
        ablated_r2s.append(r2)

    mean_ablated = float(np.mean(ablated_r2s))
    delta = baseline_r2 - mean_ablated

    return {
        "stage2_ablated_r2": mean_ablated,
        "stage2_delta": delta,
        "stage2_passes": delta >= STAGE2_DELTA_THRESHOLD,
        "n_conditioning_dims": len(cond_dims[:MAX_COND_DIMS]),
    }


def classify_dimension(stage1_result, stage2_result):
    """Classify a hidden dimension based on two-stage results.

    Returns: "MANDATORY", "ZOMBIE", or "INDIRECT"
    """
    s1_pass = stage1_result["stage1_passes"]
    s2_pass = stage2_result["stage2_passes"]

    if not s1_pass:
        return "ZOMBIE"
    elif s1_pass and s2_pass:
        return "MANDATORY"
    else:  # s1_pass and not s2_pass
        return "INDIRECT"


# ---------------------------------------------------------------------------
# Full two-stage classification
# ---------------------------------------------------------------------------

def run_twostage_classification(hidden_states, probe_target,
                                 probe_name="probe",
                                 dims_to_test=None,
                                 conditioning_sets=None,
                                 output_dir=None):
    """Run two-stage ablation classification across hidden dimensions.

    Args:
        hidden_states: (T, D) array
        probe_target: (T,) array
        probe_name: label for the probe target
        dims_to_test: list of dim indices (default: all)
        conditioning_sets: from correlation_structure

    Returns:
        dict with per-dimension results and summary
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

    print("\nTwo-stage classification: {} dims, probe={}".format(
        len(dims_to_test), probe_name))

    dim_results = []
    counts = {"MANDATORY": 0, "ZOMBIE": 0, "INDIRECT": 0}

    for i, dim in enumerate(dims_to_test):
        if (i + 1) % 20 == 0 or i == 0:
            print("  classifying dim {}/{} ({})".format(
                i + 1, len(dims_to_test), dim))

        s1 = stage1_unconditional(hidden_states, probe_target, dim)
        s2 = stage2_conditional(hidden_states, probe_target, dim,
                                conditioning_sets)
        classification = classify_dimension(s1, s2)
        counts[classification] += 1

        dim_results.append({
            "dim": dim,
            "classification": classification,
            **s1,
            **s2,
        })

    # Summary
    total = len(dims_to_test)
    summary = {
        "probe_name": probe_name,
        "n_dims_tested": total,
        "n_mandatory": counts["MANDATORY"],
        "n_zombie": counts["ZOMBIE"],
        "n_indirect": counts["INDIRECT"],
        "pct_mandatory": counts["MANDATORY"] / total * 100 if total else 0,
        "pct_zombie": counts["ZOMBIE"] / total * 100 if total else 0,
        "pct_indirect": counts["INDIRECT"] / total * 100 if total else 0,
        "stage1_delta_threshold": STAGE1_DELTA_THRESHOLD,
        "stage2_delta_threshold": STAGE2_DELTA_THRESHOLD,
    }

    output = {
        "summary": summary,
        "dim_results": dim_results,
    }

    out_file = out_path / "twostage_classification_{}.json".format(probe_name)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print report
    print("\n" + "=" * 70)
    print("PHASE 5C: TWO-STAGE CLASSIFICATION — {}".format(probe_name))
    print("=" * 70)
    print("\nDimensions tested: {}".format(total))
    print("  MANDATORY: {:>4} ({:.1f}%)".format(
        counts["MANDATORY"], summary["pct_mandatory"]))
    print("  ZOMBIE:    {:>4} ({:.1f}%)".format(
        counts["ZOMBIE"], summary["pct_zombie"]))
    print("  INDIRECT:  {:>4} ({:.1f}%)".format(
        counts["INDIRECT"], summary["pct_indirect"]))

    if counts["INDIRECT"] > 0:
        print("\n--- INDIRECT dimensions (passed Stage 1, failed Stage 2) ---")
        print("{:<6} {:>10} {:>10} {:>10} {:>10}".format(
            "Dim", "S1 delta", "S2 delta", "S1 R2abl", "S2 R2abl"))
        print("-" * 55)
        for r in dim_results:
            if r["classification"] == "INDIRECT":
                print("{:<6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    r["dim"], r["stage1_delta"], r["stage2_delta"],
                    r["stage1_ablated_r2"], r["stage2_ablated_r2"]))

    print("\nResults saved: {}".format(out_file))
    return output


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running two-stage classification demo...")
    rng = np.random.RandomState(42)
    D, T = 32, 1000

    # Hidden states with correlation structure
    H = rng.randn(T, D).astype(np.float32)
    # Dims 1-4 correlated with dim 0
    for i in range(1, 5):
        H[:, i] = 0.6 * H[:, 0] + 0.4 * rng.randn(T)

    # Probe target depends directly on dim 0, indirectly on 1-4
    y = 0.8 * H[:, 0] + 0.2 * rng.randn(T)

    # Build conditioning sets
    cond_sets = {i: [] for i in range(D)}
    cond_sets[0] = [1, 2, 3, 4]
    for i in range(1, 5):
        cond_sets[i] = [0, *[j for j in range(1, 5) if j != i]]

    results = run_twostage_classification(
        H, y, probe_name="demo_target",
        dims_to_test=list(range(8)),
        conditioning_sets=cond_sets)

    print("\nExpected: dim 0 = MANDATORY, dims 1-4 = INDIRECT, dims 5-7 = ZOMBIE")
