"""
run_p6_dual_arch.py

Phase 6-2: Run Valdez Active Emotion with Dual Architecture + PySR

Runs LSTM, MLP-200ms, and PySR simultaneously on Valdez et al. 2022 data.

GOVERNANCE PROTOCOL:
    1. Check consent_audit FIRST — BLOCKED datasets cannot proceed
    2. Train ALL architectures BEFORE analysing ANY results
       (prevents psychological attachment to early results biasing
       threshold choices or architecture preferences)
    3. Apply Priority 3 thresholds (50-seed baseline variance)
    4. Apply Priority 5 two-stage ablation
    5. Compute Jaccard overlap between architecture mandatory sets
    6. Test primary prediction: active emotion task variable is
       mandatory in >= 50% of patients

OUTPUT: results/council_controls/phase6_valdez/

No external dependencies beyond stdlib + json + dataclasses.
Actual model training requires numpy, torch, sklearn at runtime.

Usage:
    python -m descartes.council_controls.priority6_valdez.run_p6_dual_arch \\
        --data_dir data/valdez_osf_nf7s8
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

from ..config_council import QAP


# ---------------------------------------------------------------------------
# Governance gate
# ---------------------------------------------------------------------------

def check_consent_gate(dataset_name: str = "Valdez OSF nf7s8",
                       audit_path: str = "results/governance/consent_audit.json"
                       ) -> Tuple[bool, str]:
    """Check consent audit before proceeding. BLOCKING for human data.

    Returns (can_proceed, status_message).
    """
    if not os.path.exists(audit_path):
        return False, (
            "BLOCKED: No consent audit found at {}. "
            "Run priority0b_governance.consent_audit first.".format(audit_path)
        )

    with open(audit_path) as f:
        audits = json.load(f)

    for a in audits:
        if a['dataset_name'] == dataset_name:
            status = a.get('governance_status', 'UNCHECKED')
            if status in ('CLEARED', 'CAVEAT'):
                msg = "CONSENT {}: {} — proceeding with caveats logged".format(
                    status, a.get('justification', 'see audit'))
                return True, msg
            else:
                return False, (
                    "BLOCKED: Dataset '{}' consent status is {}. "
                    "Cannot proceed with human data analysis.".format(
                        dataset_name, status))

    return False, (
        "BLOCKED: Dataset '{}' not found in consent audit. "
        "Add to DATASETS_TO_AUDIT and re-run consent_audit.".format(dataset_name))


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of two-stage ablation for one architecture, one patient."""
    patient_id: str
    architecture: str
    seed: int
    stage1_survivors: List[str]
    stage2_mandatory: List[str]
    baseline_r: float
    final_r: float


@dataclass
class PatientArchResult:
    """Aggregated results for one patient across all seeds for one architecture."""
    patient_id: str
    architecture: str
    n_seeds: int
    mandatory_variables: List[str]  # survived in >= baseline_mandatory_threshold seeds
    mandatory_counts: Dict[str, int]  # variable -> count of seeds where mandatory
    mean_baseline_r: float
    std_baseline_r: float


@dataclass
class DualArchResult:
    """Cross-architecture comparison for one patient."""
    patient_id: str
    lstm_mandatory: List[str]
    mlp_mandatory: List[str]
    pysr_mandatory: List[str]
    jaccard_lstm_mlp: float
    jaccard_lstm_pysr: float
    jaccard_mlp_pysr: float
    consensus_mandatory: List[str]  # in >= 2 of 3 architectures
    active_emotion_variable_mandatory: bool


@dataclass
class Phase6Result:
    """Full Phase 6 result."""
    timestamp: str
    consent_status: str
    n_patients: int
    architectures: List[str]
    patient_results: List[DualArchResult]
    # Primary prediction
    n_patients_with_active_emotion_mandatory: int
    fraction_patients_active_emotion: float
    primary_prediction_met: bool
    governance_caveats: List[str]


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Seed-level ablation (stub — real impl uses P3 and P5 infrastructure)
# ---------------------------------------------------------------------------

def run_single_seed_ablation(patient_data, architecture: str, seed: int,
                             variable_names: List[str]) -> AblationResult:
    """Run two-stage ablation for one seed, one architecture, one patient.

    This is a STUB that defines the interface. Real implementation calls:
        - priority3_baseline_variance for 50-seed training
        - priority5_twostage_ablation for the ablation protocol

    Parameters
    ----------
    patient_data : ValdezPatientData
        Patient data from valdez_loader.
    architecture : str
        One of 'LSTM', 'MLP-200ms', 'PySR'.
    seed : int
        Random seed for this run.
    variable_names : list of str
        Names of input variables to ablate.

    Returns
    -------
    AblationResult
    """
    # STUB: In production, this calls the actual training + ablation pipeline
    return AblationResult(
        patient_id=patient_data.patient_id if hasattr(patient_data, 'patient_id') else 'unknown',
        architecture=architecture,
        seed=seed,
        stage1_survivors=list(variable_names),
        stage2_mandatory=[],
        baseline_r=0.0,
        final_r=0.0,
    )


def aggregate_seeds(ablation_results: List[AblationResult],
                    n_seeds: int,
                    threshold: float = None) -> PatientArchResult:
    """Aggregate ablation results across seeds for one patient+architecture.

    A variable is MANDATORY if it survives ablation in >= threshold fraction
    of seeds.
    """
    if threshold is None:
        threshold = QAP.baseline_mandatory_threshold

    if not ablation_results:
        return PatientArchResult(
            patient_id='unknown', architecture='unknown',
            n_seeds=0, mandatory_variables=[], mandatory_counts={},
            mean_baseline_r=0.0, std_baseline_r=0.0)

    patient_id = ablation_results[0].patient_id
    architecture = ablation_results[0].architecture

    # Count how often each variable is mandatory across seeds
    variable_counts = {}
    baseline_rs = []
    for res in ablation_results:
        baseline_rs.append(res.baseline_r)
        for var in res.stage2_mandatory:
            variable_counts[var] = variable_counts.get(var, 0) + 1

    min_count = int(threshold * n_seeds)
    mandatory = [var for var, count in variable_counts.items()
                 if count >= min_count]

    mean_r = sum(baseline_rs) / len(baseline_rs) if baseline_rs else 0.0
    std_r = 0.0
    if len(baseline_rs) > 1:
        mean_val = mean_r
        variance = sum((r - mean_val) ** 2 for r in baseline_rs) / (len(baseline_rs) - 1)
        std_r = variance ** 0.5

    return PatientArchResult(
        patient_id=patient_id,
        architecture=architecture,
        n_seeds=n_seeds,
        mandatory_variables=sorted(mandatory),
        mandatory_counts=variable_counts,
        mean_baseline_r=round(mean_r, QAP.float_precision),
        std_baseline_r=round(std_r, QAP.float_precision),
    )


# ---------------------------------------------------------------------------
# Cross-architecture comparison
# ---------------------------------------------------------------------------

ACTIVE_EMOTION_MARKERS = [
    "emotion_category", "valence", "arousal", "task_engagement",
    "reaction_time", "response_accuracy", "emotional_intensity",
]


def compare_architectures(lstm_result: PatientArchResult,
                          mlp_result: PatientArchResult,
                          pysr_result: PatientArchResult) -> DualArchResult:
    """Compare mandatory variable sets across three architectures."""
    lstm_set = set(lstm_result.mandatory_variables)
    mlp_set = set(mlp_result.mandatory_variables)
    pysr_set = set(pysr_result.mandatory_variables)

    j_lstm_mlp = jaccard(lstm_set, mlp_set)
    j_lstm_pysr = jaccard(lstm_set, pysr_set)
    j_mlp_pysr = jaccard(mlp_set, pysr_set)

    # Consensus: appears in >= 2 of 3 architectures
    all_vars = lstm_set | mlp_set | pysr_set
    consensus = []
    for var in all_vars:
        count = sum([
            var in lstm_set,
            var in mlp_set,
            var in pysr_set,
        ])
        if count >= 2:
            consensus.append(var)

    # Check if any active emotion marker is in consensus mandatory set
    active_mandatory = any(
        marker in consensus for marker in ACTIVE_EMOTION_MARKERS
    )

    return DualArchResult(
        patient_id=lstm_result.patient_id,
        lstm_mandatory=sorted(lstm_set),
        mlp_mandatory=sorted(mlp_set),
        pysr_mandatory=sorted(pysr_set),
        jaccard_lstm_mlp=round(j_lstm_mlp, QAP.float_precision),
        jaccard_lstm_pysr=round(j_lstm_pysr, QAP.float_precision),
        jaccard_mlp_pysr=round(j_mlp_pysr, QAP.float_precision),
        consensus_mandatory=sorted(consensus),
        active_emotion_variable_mandatory=active_mandatory,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase6(data_dir: str = "data/valdez_osf_nf7s8",
               output_dir: str = None,
               consent_audit_path: str = "results/governance/consent_audit.json"
               ) -> Optional[Phase6Result]:
    """Execute the full Phase 6 pipeline.

    Protocol:
        1. Consent gate (BLOCKING)
        2. Load data
        3. Train ALL architectures for ALL patients (no peeking)
        4. Ablation across 50-seed baseline per architecture
        5. Cross-architecture Jaccard comparison
        6. Test primary prediction
    """
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase6_valdez")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Consent gate ---
    can_proceed, consent_msg = check_consent_gate(
        audit_path=consent_audit_path)
    print("\n" + "=" * 70)
    print("PHASE 6: VALDEZ ACTIVE EMOTION — DUAL ARCHITECTURE")
    print("=" * 70)
    print("Consent check: {}".format(consent_msg))

    if not can_proceed:
        print("\nPIPELINE HALTED: Consent gate not passed.")
        # Save blocked status
        blocked_result = {
            'timestamp': datetime.now().isoformat(),
            'status': 'BLOCKED',
            'reason': consent_msg,
        }
        with open(out_path / "phase6_BLOCKED.json", 'w') as f:
            json.dump(blocked_result, f, indent=2)
        return None

    # --- Step 2: Load data ---
    from .valdez_loader import load_valdez_from_osf, print_dataset_summary
    dataset = load_valdez_from_osf(data_dir)
    print_dataset_summary(dataset)

    if dataset.n_patients == 0:
        print("WARNING: No patients loaded. Check data directory.")
        return None

    # --- Step 3 + 4: Train ALL architectures, THEN analyse ---
    # CRITICAL: Train all before analysing any, to prevent psychological
    # attachment to early results biasing later analysis decisions.

    architectures = list(QAP.valdez_architectures)
    n_seeds = QAP.baseline_seed_count
    variable_names = list(QAP.bridge_shared_variables) + ACTIVE_EMOTION_MARKERS

    print("\nTraining {} architectures x {} seeds x {} patients...".format(
        len(architectures), n_seeds, dataset.n_patients))
    print("PROTOCOL: All training completes before any analysis begins.")

    # Phase A: Train everything (store results, don't look)
    all_ablation_results = {}  # (patient_id, arch) -> list of AblationResult
    for patient in dataset.patients:
        for arch in architectures:
            key = (patient.patient_id, arch)
            all_ablation_results[key] = []
            for seed in range(n_seeds):
                result = run_single_seed_ablation(
                    patient, arch, seed + QAP.random_seed_base,
                    variable_names)
                all_ablation_results[key].append(result)

    print("Training complete. Beginning analysis phase.\n")

    # Phase B: Analyse (now allowed to look at results)
    patient_results = []
    n_active_mandatory = 0

    for patient in dataset.patients:
        arch_results = {}
        for arch in architectures:
            key = (patient.patient_id, arch)
            agg = aggregate_seeds(
                all_ablation_results[key], n_seeds)
            arch_results[arch] = agg

        # Cross-architecture comparison
        dual = compare_architectures(
            arch_results.get("LSTM", PatientArchResult(
                patient.patient_id, "LSTM", 0, [], {}, 0, 0)),
            arch_results.get("MLP-200ms", PatientArchResult(
                patient.patient_id, "MLP-200ms", 0, [], {}, 0, 0)),
            arch_results.get("PySR", PatientArchResult(
                patient.patient_id, "PySR", 0, [], {}, 0, 0)),
        )
        patient_results.append(dual)

        if dual.active_emotion_variable_mandatory:
            n_active_mandatory += 1

        # Print per-patient summary
        print("Patient {}: consensus mandatory = {}".format(
            patient.patient_id, dual.consensus_mandatory))
        print("  Jaccard LSTM/MLP={:.3f}  LSTM/PySR={:.3f}  MLP/PySR={:.3f}".format(
            dual.jaccard_lstm_mlp, dual.jaccard_lstm_pysr, dual.jaccard_mlp_pysr))
        print("  Active emotion mandatory: {}".format(
            dual.active_emotion_variable_mandatory))

    # --- Step 6: Primary prediction ---
    fraction = n_active_mandatory / dataset.n_patients if dataset.n_patients > 0 else 0.0
    prediction_met = fraction >= QAP.valdez_mandatory_patient_fraction

    governance_caveats = [
        "AI quality panel recommendation only — not yet validated by external human reviewer",
        "Experimental design not reviewed by patient/public advisors",
    ]

    result = Phase6Result(
        timestamp=datetime.now().isoformat(),
        consent_status=consent_msg,
        n_patients=dataset.n_patients,
        architectures=architectures,
        patient_results=patient_results,
        n_patients_with_active_emotion_mandatory=n_active_mandatory,
        fraction_patients_active_emotion=round(fraction, QAP.float_precision),
        primary_prediction_met=prediction_met,
        governance_caveats=governance_caveats,
    )

    # --- Save results ---
    result_dict = asdict(result)
    with open(out_path / "phase6_results.json", 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("PHASE 6 SUMMARY")
    print("=" * 70)
    print("Patients analysed:                  {}".format(result.n_patients))
    print("Architectures:                      {}".format(
        ", ".join(result.architectures)))
    print("Seeds per architecture:             {}".format(n_seeds))
    print("Active emotion mandatory in:        {}/{} patients ({:.1%})".format(
        n_active_mandatory, dataset.n_patients, fraction))
    print("Primary prediction (>= {:.0%}):      {}".format(
        QAP.valdez_mandatory_patient_fraction,
        "MET" if prediction_met else "NOT MET"))
    print("\nGovernance caveats:")
    for c in governance_caveats:
        print("  - {}".format(c))
    print("\nResults saved: {}".format(out_path))
    print("=" * 70)

    return result


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/valdez_osf_nf7s8"
    run_phase6(data_dir=data_dir)
