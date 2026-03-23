"""
run_p4_pipeline.py

Phase 4: Full Gated Pipeline

Orchestrates the three-step gated seizure analysis:
  Step 1: check_ictal_epochs  — Verify ictal data exists (consent-gated)
  Step 2: stability_gate      — GO/NO-GO partition stability check
  Step 3: differential_analysis — Mandatory vs zombie during seizure

Each step is a gate: if it fails, the pipeline halts with a clear reason.
This prevents invalid downstream analysis.

Usage:
    python -m descartes.council_controls.priority4_seizure_aura.run_p4_pipeline [nwb_path]
"""

import json
import sys
import time
from pathlib import Path

from .check_ictal_epochs import check_ictal
from .stability_gate import run_stability_gate
from .differential_analysis import run_differential

RESULTS_DIR = Path("results/council_controls/phase4_seizure_aura")


def run_phase4_pipeline(dandiset_path=None, patient_data_path=None,
                        seizure_data_path=None, r2_threshold=0.15,
                        output_dir=None):
    """Run the full Phase 4 gated pipeline.

    Args:
        dandiset_path: Path to local NWB files for 000576.
        patient_data_path: Path to patient R2 data for stability gate.
        seizure_data_path: Path to patient seizure R2 data.
        r2_threshold: R2 threshold for mandatory/zombie classification.
        output_dir: Output directory for results.

    Returns:
        dict with pipeline results and final status.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipeline_log = []

    print("=" * 70)
    print("PHASE 4: SEIZURE/AURA GATED PIPELINE")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Check for ictal data (consent-gated)
    # -----------------------------------------------------------------------
    print("\n--- STEP 1: Ictal Epoch Check ---")
    ictal_result = check_ictal(dandiset_path=dandiset_path,
                                output_dir=str(out_path))

    pipeline_log.append({
        "step": 1,
        "name": "check_ictal_epochs",
        "status": "PASS" if ictal_result["ictal_found"] else "HALT",
        "consent_status": ictal_result["consent_status"],
        "ictal_found": ictal_result["ictal_found"],
    })

    if ictal_result.get("method") == "blocked_by_governance":
        print("\nPIPELINE HALTED at Step 1: Governance block.")
        return _save_pipeline_result(out_path, "HALTED_GOVERNANCE",
                                     pipeline_log, t0)

    if not ictal_result["ictal_found"]:
        print("\nPIPELINE HALTED at Step 1: No ictal data found.")
        print("Differential seizure analysis requires ictal epochs.")
        return _save_pipeline_result(out_path, "HALTED_NO_ICTAL",
                                     pipeline_log, t0)

    # -----------------------------------------------------------------------
    # Step 2: Stability gate
    # -----------------------------------------------------------------------
    print("\n--- STEP 2: Stability Gate ---")
    gate_result = run_stability_gate(data_path=patient_data_path,
                                      threshold=r2_threshold,
                                      output_dir=str(out_path))

    pipeline_log.append({
        "step": 2,
        "name": "stability_gate",
        "status": gate_result["decision"],
        "n_passing": gate_result["n_passing"],
        "n_patients": gate_result["n_patients"],
    })

    if gate_result["decision"] == "NO-GO":
        print("\nPIPELINE HALTED at Step 2: Partition unstable.")
        return _save_pipeline_result(out_path, "HALTED_UNSTABLE",
                                     pipeline_log, t0)

    # -----------------------------------------------------------------------
    # Step 3: Differential analysis
    # -----------------------------------------------------------------------
    print("\n--- STEP 3: Differential Analysis ---")
    diff_result = run_differential(data_path=seizure_data_path,
                                    output_dir=str(out_path))

    pipeline_log.append({
        "step": 3,
        "name": "differential_analysis",
        "status": diff_result["overall_verdict"],
        "outcome_counts": diff_result["outcome_counts"],
    })

    final_status = "COMPLETE_{}".format(diff_result["overall_verdict"])
    return _save_pipeline_result(out_path, final_status, pipeline_log, t0)


def _save_pipeline_result(out_path, status, pipeline_log, t0):
    """Save pipeline result and return summary."""
    elapsed = time.time() - t0
    result = {
        "pipeline": "Phase 4: Seizure/Aura",
        "final_status": status,
        "elapsed_seconds": round(elapsed, 1),
        "steps": pipeline_log,
    }

    out_file = out_path / "pipeline_result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE RESULT: {}".format(status))
    print("Elapsed: {:.1f}s".format(elapsed))
    print("Steps completed: {}".format(len(pipeline_log)))
    for step in pipeline_log:
        print("  Step {}: {} -> {}".format(
            step["step"], step["name"], step["status"]))
    print("Saved: {}".format(out_file))
    print("=" * 70)

    return result


if __name__ == "__main__":
    nwb_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_phase4_pipeline(dandiset_path=nwb_path)
