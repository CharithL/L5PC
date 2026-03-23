"""
stability_gate.py

Phase 4B: GO/NO-GO Stability Gate

Before running seizure differential analysis, verify that the
mandatory/zombie partition is stable between clean task data and
pre-ictal baseline. If the partition is unstable, any differential
result during seizure is uninterpretable.

GO criteria (ALL must be met):
  - Jaccard similarity of mandatory sets > 0.7
  - Spearman rank correlation of R2 values > 0.8
  - Both criteria met for >= 3 patients

If NO-GO: halt Phase 4 and report instability.

Usage:
    python -m descartes.council_controls.priority4_seizure_aura.stability_gate
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

RESULTS_DIR = Path("results/council_controls/phase4_seizure_aura")

# GO/NO-GO thresholds
JACCARD_THRESHOLD = 0.7
RANK_CORR_THRESHOLD = 0.8
MIN_PATIENTS_GO = 3


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def rank_correlation(r2_dict_a, r2_dict_b):
    """Spearman rank correlation between two R2 dictionaries.

    Only uses variables present in both dictionaries.
    """
    common_keys = sorted(set(r2_dict_a.keys()) & set(r2_dict_b.keys()))
    if len(common_keys) < 3:
        return 0.0, 1.0  # Not enough variables

    vals_a = [r2_dict_a[k] for k in common_keys]
    vals_b = [r2_dict_b[k] for k in common_keys]

    rho, pval = sp_stats.spearmanr(vals_a, vals_b)
    return float(rho), float(pval)


def classify_variables(r2_dict, threshold):
    """Classify variables into mandatory/zombie based on threshold.

    Args:
        r2_dict: {var_name: r2_value}
        threshold: R2 above this = mandatory

    Returns:
        (mandatory_set, zombie_set)
    """
    mandatory = set()
    zombie = set()
    for var, r2 in r2_dict.items():
        if r2 > threshold:
            mandatory.add(var)
        else:
            zombie.add(var)
    return mandatory, zombie


def evaluate_patient(clean_r2, preictal_r2, threshold):
    """Evaluate stability for a single patient.

    Args:
        clean_r2: {var_name: r2_value} from clean task epochs
        preictal_r2: {var_name: r2_value} from pre-ictal baseline
        threshold: R2 threshold for mandatory classification

    Returns:
        dict with stability metrics
    """
    mand_clean, zombie_clean = classify_variables(clean_r2, threshold)
    mand_preictal, zombie_preictal = classify_variables(preictal_r2, threshold)

    jacc = jaccard_similarity(mand_clean, mand_preictal)
    rho, rho_pval = rank_correlation(clean_r2, preictal_r2)

    passes_jaccard = jacc >= JACCARD_THRESHOLD
    passes_rank = rho >= RANK_CORR_THRESHOLD
    passes = passes_jaccard and passes_rank

    return {
        "mandatory_clean": sorted(mand_clean),
        "mandatory_preictal": sorted(mand_preictal),
        "zombie_clean": sorted(zombie_clean),
        "zombie_preictal": sorted(zombie_preictal),
        "jaccard": jacc,
        "rank_corr": rho,
        "rank_corr_pval": rho_pval,
        "passes_jaccard": passes_jaccard,
        "passes_rank": passes_rank,
        "passes": passes,
    }


def load_patient_data(data_path=None):
    """Load per-patient R2 data from clean and pre-ictal epochs.

    In the full pipeline, this reads from probe results. Here we provide
    synthetic data to demonstrate the stability gate logic.
    """
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.load(f)

    # Synthetic demonstration data
    rng = np.random.RandomState(123)
    targets = [
        "firing_rate", "spike_count", "isi_cv", "burst_index",
        "population_sync", "oscillation_power", "phase_coherence",
        "granger_cause", "transfer_entropy", "mutual_info",
    ]

    patients = {}
    for pid in ["P01", "P02", "P03", "P04", "P05"]:
        base_r2 = {t: rng.uniform(0.05, 0.5) for t in targets}
        # Pre-ictal: add noise but keep similar structure (stable patients)
        # Except P05 which is unstable
        noise_scale = 0.3 if pid == "P05" else 0.05
        preictal_r2 = {
            t: max(0, v + rng.normal(0, noise_scale))
            for t, v in base_r2.items()
        }
        patients[pid] = {
            "clean_r2": base_r2,
            "preictal_r2": preictal_r2,
        }

    return patients


def run_stability_gate(data_path=None, threshold=0.15, output_dir=None):
    """Run the GO/NO-GO stability gate across patients.

    Args:
        data_path: Path to patient R2 data JSON.
        threshold: R2 threshold for mandatory/zombie classification.
        output_dir: Output directory.

    Returns:
        dict: {"decision": "GO"/"NO-GO", "patient_results": {...}, ...}
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patients = load_patient_data(data_path)
    patient_results = {}
    n_passing = 0

    for pid, pdata in patients.items():
        result = evaluate_patient(
            pdata["clean_r2"], pdata["preictal_r2"], threshold)
        patient_results[pid] = result
        if result["passes"]:
            n_passing += 1

    decision = "GO" if n_passing >= MIN_PATIENTS_GO else "NO-GO"

    gate_result = {
        "decision": decision,
        "n_patients": len(patients),
        "n_passing": n_passing,
        "min_required": MIN_PATIENTS_GO,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "rank_corr_threshold": RANK_CORR_THRESHOLD,
        "r2_threshold": threshold,
        "patient_results": patient_results,
    }

    out_file = out_path / "stability_gate.json"
    with open(out_file, "w") as f:
        json.dump(gate_result, f, indent=2)

    # Print report
    print("\n" + "=" * 70)
    print("PHASE 4B: STABILITY GATE (GO/NO-GO)")
    print("=" * 70)
    print("\nDecision: {}".format(decision))
    print("Patients passing: {}/{}  (require >= {})".format(
        n_passing, len(patients), MIN_PATIENTS_GO))
    print("\nThresholds: Jaccard >= {}, Rank corr >= {}".format(
        JACCARD_THRESHOLD, RANK_CORR_THRESHOLD))

    print("\n{:<8} {:>8} {:>8} {:>6} {:>6} {:>8}".format(
        "Patient", "Jaccard", "RankR", "Jacc?", "Rank?", "Overall"))
    print("-" * 55)

    for pid in sorted(patient_results.keys()):
        r = patient_results[pid]
        print("{:<8} {:>8.3f} {:>8.3f} {:>6} {:>6} {:>8}".format(
            pid, r["jaccard"], r["rank_corr"],
            "PASS" if r["passes_jaccard"] else "FAIL",
            "PASS" if r["passes_rank"] else "FAIL",
            "PASS" if r["passes"] else "FAIL"))

    if decision == "NO-GO":
        print("\n** NO-GO: Mandatory/zombie partition is unstable across "
              "conditions. Differential seizure analysis would be "
              "uninterpretable. Phase 4 halted. **")
    else:
        print("\nGO: Partition stable. Proceeding to differential analysis.")

    print("\nResults saved: {}".format(out_file))
    return gate_result


if __name__ == "__main__":
    run_stability_gate()
