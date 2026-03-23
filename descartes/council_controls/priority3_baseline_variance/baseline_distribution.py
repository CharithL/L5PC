"""
baseline_distribution.py

Phase 3B: Baseline Distribution Statistics and Permutation Thresholds

Consumes the 50-seed R2 distributions from run_50_seeds.py and computes:
- Full distribution statistics (mean, std, median, percentiles, skewness)
- Permutation threshold = 97.5th percentile of untrained R2
- Heavy-tail flags for distributions where skewness > 1.0

The 97.5th percentile becomes the new bar: any trained R2 that does not
exceed this threshold is indistinguishable from what an untrained network
can achieve by chance.

Usage:
    python -m descartes.council_controls.priority3_baseline_variance.baseline_distribution
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats


RESULTS_DIR = Path("results/council_controls/phase3_baseline_variance")
THRESHOLD_PERCENTILE = 97.5
HEAVY_TAIL_SKEWNESS = 1.0


def compute_distribution_stats(r2_values):
    """Compute full distribution statistics for a list of R2 values.

    Returns:
        dict with keys: mean, std, median, p2_5, p25, p75, p97_5,
                        min, max, skewness, kurtosis, threshold,
                        heavy_tailed
    """
    arr = np.array(r2_values)
    if len(arr) == 0:
        return {"n": 0, "error": "empty distribution"}

    skew = float(sp_stats.skew(arr))
    kurt = float(sp_stats.kurtosis(arr))
    threshold = float(np.percentile(arr, THRESHOLD_PERCENTILE))

    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p97_5": threshold,
        "skewness": skew,
        "kurtosis": kurt,
        "threshold": threshold,
        "heavy_tailed": bool(abs(skew) > HEAVY_TAIL_SKEWNESS),
    }


def analyze_distributions(raw_path=None, output_dir=None):
    """Load raw 50-seed results and compute per-variable statistics.

    Returns:
        dict: {circuit: {arch: {target: stats_dict}}}
    """
    if raw_path is None:
        raw_path = RESULTS_DIR / "raw_50seed_r2.json"
    if output_dir is None:
        output_dir = RESULTS_DIR

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(raw_path) as f:
        raw = json.load(f)

    all_stats = {}
    heavy_tail_warnings = []

    for cid, arch_dict in raw.items():
        all_stats[cid] = {}
        for arch, target_dict in arch_dict.items():
            all_stats[cid][arch] = {}
            for target, r2_list in target_dict.items():
                s = compute_distribution_stats(r2_list)
                all_stats[cid][arch][target] = s

                if s.get("heavy_tailed"):
                    heavy_tail_warnings.append({
                        "circuit": cid,
                        "architecture": arch,
                        "target": target,
                        "skewness": s["skewness"],
                        "note": (
                            "Heavy-tailed baseline distribution (|skew| > {:.1f}). "
                            "The 97.5th percentile threshold may be unreliable. "
                            "Consider using a bootstrap CI or non-parametric test "
                            "instead of a fixed percentile."
                        ).format(HEAVY_TAIL_SKEWNESS),
                    })

    # Save statistics
    stats_file = out_path / "baseline_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Save thresholds as a flat lookup table
    thresholds = {}
    for cid, arch_dict in all_stats.items():
        thresholds[cid] = {}
        for arch, target_dict in arch_dict.items():
            thresholds[cid][arch] = {}
            for target, s in target_dict.items():
                thresholds[cid][arch][target] = s.get("threshold", None)

    thresh_file = out_path / "permutation_thresholds.json"
    with open(thresh_file, "w") as f:
        json.dump(thresholds, f, indent=2)

    # Save heavy-tail warnings
    if heavy_tail_warnings:
        warn_file = out_path / "heavy_tail_warnings.json"
        with open(warn_file, "w") as f:
            json.dump(heavy_tail_warnings, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 3B: BASELINE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for cid in sorted(all_stats.keys()):
        for arch in sorted(all_stats[cid].keys()):
            print("\n--- {} x {} ---".format(cid, arch))
            print("{:<22} {:>8} {:>8} {:>8} {:>8} {:>6}".format(
                "Target", "Mean", "Std", "p97.5", "Skew", "HeavyT"))
            print("-" * 70)
            for target in sorted(all_stats[cid][arch].keys()):
                s = all_stats[cid][arch][target]
                flag = "YES" if s.get("heavy_tailed") else ""
                print("{:<22} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.2f} {:>6}".format(
                    target, s["mean"], s["std"], s["threshold"],
                    s["skewness"], flag))

    if heavy_tail_warnings:
        print("\n\nWARNING: {} heavy-tailed distributions detected.".format(
            len(heavy_tail_warnings)))
        print("See: {}".format(out_path / "heavy_tail_warnings.json"))

    print("\nThresholds saved: {}".format(thresh_file))
    print("Full statistics: {}".format(stats_file))

    return all_stats, thresholds


def load_thresholds(path=None):
    """Convenience loader for downstream use (e.g., retrofit_all_circuits)."""
    if path is None:
        path = RESULTS_DIR / "permutation_thresholds.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    analyze_distributions()
