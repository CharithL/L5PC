"""
retrofit_all_circuits.py

Phase 3C: Retrofit Previously Mandatory Variables Against Permutation Thresholds

For each variable previously classified as "mandatory" in the DESCARTES
pipeline, compare its trained R2 against the new permutation threshold
(97.5th percentile of the 50-seed untrained distribution).

Reports:
- How many previously mandatory variables fail the new threshold (deflation count)
- Which variables survive as genuinely mandatory
- Which should be reclassified as "zombie" (trained R2 <= untrained p97.5)

This is the honest-broker step: it may invalidate previously reported results.

Usage:
    python -m descartes.council_controls.priority3_baseline_variance.retrofit_all_circuits
"""

import json
from pathlib import Path

import numpy as np

from .baseline_distribution import load_thresholds

RESULTS_DIR = Path("results/council_controls/phase3_baseline_variance")


# ---------------------------------------------------------------------------
# Simulated prior results (in full pipeline, loaded from trained probe logs)
# ---------------------------------------------------------------------------

def load_prior_mandatory_results(path=None):
    """Load previously reported mandatory variable R2 values.

    In the full pipeline, this reads from the trained probe result logs.
    Here we provide a placeholder that returns synthetic trained R2 values
    to demonstrate the retrofit logic.
    """
    if path is not None and Path(path).exists():
        with open(path) as f:
            return json.load(f)

    # Placeholder: simulate trained R2 values for demonstration
    # Some will exceed threshold (genuinely mandatory), some will not (zombies)
    rng = np.random.RandomState(42)
    from .run_50_seeds import CIRCUITS, ARCHITECTURES, PROBE_TARGETS

    prior = {}
    for cid in CIRCUITS:
        prior[cid] = {}
        for arch in ARCHITECTURES:
            prior[cid][arch] = {}
            for target in PROBE_TARGETS:
                # Mix of moderate and high R2 — some will be deflated
                r2 = rng.uniform(0.01, 0.45)
                prior[cid][arch][target] = {
                    "r2_trained": float(r2),
                    "previously_classified": "mandatory",
                }
    return prior


# ---------------------------------------------------------------------------
# Retrofit logic
# ---------------------------------------------------------------------------

def retrofit(prior_results, thresholds):
    """Compare trained R2 against permutation thresholds.

    Returns:
        dict with retrofit results and summary counts
    """
    retrofitted = {}
    total_mandatory = 0
    total_deflated = 0
    total_survived = 0

    for cid in prior_results:
        retrofitted[cid] = {}
        for arch in prior_results[cid]:
            retrofitted[cid][arch] = {}
            for target, info in prior_results[cid][arch].items():
                r2_trained = info["r2_trained"]
                threshold = None

                # Look up threshold
                if (cid in thresholds and arch in thresholds[cid]
                        and target in thresholds[cid][arch]):
                    threshold = thresholds[cid][arch][target]

                if threshold is None:
                    new_class = "UNCHECKED"
                    exceeded = None
                    margin = None
                elif r2_trained > threshold:
                    new_class = "MANDATORY_CONFIRMED"
                    exceeded = True
                    margin = r2_trained - threshold
                    total_survived += 1
                else:
                    new_class = "ZOMBIE"
                    exceeded = False
                    margin = r2_trained - threshold
                    total_deflated += 1

                total_mandatory += 1

                retrofitted[cid][arch][target] = {
                    "r2_trained": r2_trained,
                    "threshold_p97_5": threshold,
                    "exceeds_threshold": exceeded,
                    "margin": float(margin) if margin is not None else None,
                    "previous_classification": info.get(
                        "previously_classified", "mandatory"),
                    "new_classification": new_class,
                }

    summary = {
        "total_variables_tested": total_mandatory,
        "survived_as_mandatory": total_survived,
        "deflated_to_zombie": total_deflated,
        "deflation_rate": (
            total_deflated / total_mandatory if total_mandatory > 0 else 0.0
        ),
    }

    return retrofitted, summary


def run_retrofit(prior_path=None, threshold_path=None, output_dir=None):
    """Run the full retrofit analysis."""
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(threshold_path)
    prior = load_prior_mandatory_results(prior_path)
    retrofitted, summary = retrofit(prior, thresholds)

    # Save
    retro_file = out_path / "retrofit_results.json"
    with open(retro_file, "w") as f:
        json.dump({"results": retrofitted, "summary": summary}, f, indent=2)

    # Print report
    print("\n" + "=" * 80)
    print("PHASE 3C: RETROFIT MANDATORY VARIABLES")
    print("=" * 80)
    print("\nTotal variables tested:  {}".format(summary["total_variables_tested"]))
    print("Survived as MANDATORY:  {}".format(summary["survived_as_mandatory"]))
    print("Deflated to ZOMBIE:     {}".format(summary["deflated_to_zombie"]))
    print("Deflation rate:         {:.1%}".format(summary["deflation_rate"]))

    # Detail table
    print("\n{:<6} {:<6} {:<22} {:>8} {:>8} {:>8} {:<20}".format(
        "Circ", "Arch", "Target", "R2_trn", "Thresh", "Margin", "New Class"))
    print("-" * 90)

    for cid in sorted(retrofitted.keys()):
        for arch in sorted(retrofitted[cid].keys()):
            for target in sorted(retrofitted[cid][arch].keys()):
                r = retrofitted[cid][arch][target]
                thresh_str = "{:.4f}".format(r["threshold_p97_5"]) if r["threshold_p97_5"] is not None else "N/A"
                margin_str = "{:+.4f}".format(r["margin"]) if r["margin"] is not None else "N/A"
                print("{:<6} {:<6} {:<22} {:>8.4f} {:>8} {:>8} {:<20}".format(
                    cid, arch, target, r["r2_trained"],
                    thresh_str, margin_str, r["new_classification"]))

    if summary["deflation_rate"] > 0.3:
        print("\n** HIGH DEFLATION WARNING: {:.0%} of previously mandatory "
              "variables do not exceed the untrained baseline. "
              "Pipeline claims require revision. **".format(
                  summary["deflation_rate"]))

    print("\nRetrofit results saved: {}".format(retro_file))
    return retrofitted, summary


if __name__ == "__main__":
    run_retrofit()
