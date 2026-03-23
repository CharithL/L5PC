"""
active_vs_passive.py

Phase 6-3: Direct Comparison of Active (Valdez) vs Passive (C5) Tasks

Compares mandatory variable sets between:
    - Valdez et al. 2022 (ACTIVE emotion task: recognition/discrimination)
    - C5 dataset (PASSIVE viewing of emotional stimuli)

Key question: Does active emotional engagement produce DIFFERENT mandatory
variables than passive viewing? If active task variables are mandatory in
Valdez but not in C5, this suggests task engagement modulates which neural
features are necessary for prediction.

Metrics:
    - Mandatory variable rate per dataset
    - Overlap (Jaccard) between active and passive mandatory sets
    - Variables unique to active vs unique to passive
    - Patient-level consistency

No external dependencies beyond stdlib + json + dataclasses.

Usage:
    python -m descartes.council_controls.priority6_valdez.active_vs_passive
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

from ..config_council import QAP


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetMandatorySummary:
    """Summary of mandatory variables for one dataset."""
    dataset_name: str
    task_type: str  # "active" or "passive"
    n_patients: int
    mandatory_variables_per_patient: Dict[str, List[str]]
    # Union of all mandatory variables across patients
    all_mandatory: List[str] = field(default_factory=list)
    # Variables mandatory in >= 50% of patients
    consistent_mandatory: List[str] = field(default_factory=list)
    # Mean number of mandatory variables per patient
    mean_mandatory_count: float = 0.0


@dataclass
class ActiveVsPassiveResult:
    """Comparison result between active and passive datasets."""
    timestamp: str
    active_dataset: str
    passive_dataset: str
    active_summary: DatasetMandatorySummary
    passive_summary: DatasetMandatorySummary
    # Overlap metrics
    jaccard_all: float          # Jaccard over union of all mandatory
    jaccard_consistent: float   # Jaccard over consistently mandatory
    # Unique to each
    unique_to_active: List[str]
    unique_to_passive: List[str]
    shared: List[str]
    # Rate comparison
    active_mandatory_rate: float   # mean mandatory count / total variables
    passive_mandatory_rate: float
    rate_difference: float
    # Interpretation
    interpretation: str
    governance_caveats: List[str]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_phase6_results(results_path: str = None) -> Optional[dict]:
    """Load Phase 6 Valdez results."""
    if results_path is None:
        results_path = os.path.join(
            QAP.results_dir, "phase6_valdez", "phase6_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


def load_c5_results(results_path: str = None) -> Optional[dict]:
    """Load C5 passive viewing results.

    Expected format (from a prior pipeline run):
    {
        "patients": [
            {
                "patient_id": "...",
                "mandatory_variables": ["var1", "var2", ...]
            }, ...
        ]
    }
    """
    if results_path is None:
        results_path = os.path.join(
            QAP.results_dir, "c5_passive", "c5_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def summarise_dataset(dataset_results: dict,
                      dataset_name: str,
                      task_type: str) -> DatasetMandatorySummary:
    """Extract mandatory variable summary from pipeline results."""
    patients = dataset_results.get('patient_results', [])
    if not patients:
        patients = dataset_results.get('patients', [])

    mandatory_per_patient = {}
    for p in patients:
        pid = p.get('patient_id', 'unknown')
        # Phase 6 format uses consensus_mandatory
        mandatory = p.get('consensus_mandatory',
                          p.get('mandatory_variables', []))
        mandatory_per_patient[pid] = mandatory

    n_patients = len(mandatory_per_patient)
    all_vars = set()
    for vars_list in mandatory_per_patient.values():
        all_vars.update(vars_list)

    # Count how many patients have each variable as mandatory
    var_counts = {}
    for vars_list in mandatory_per_patient.values():
        for var in vars_list:
            var_counts[var] = var_counts.get(var, 0) + 1

    # Consistent = mandatory in >= 50% of patients
    min_patients = max(1, n_patients // 2)
    consistent = [var for var, count in var_counts.items()
                  if count >= min_patients]

    # Mean count
    counts = [len(v) for v in mandatory_per_patient.values()]
    mean_count = sum(counts) / len(counts) if counts else 0.0

    return DatasetMandatorySummary(
        dataset_name=dataset_name,
        task_type=task_type,
        n_patients=n_patients,
        mandatory_variables_per_patient=mandatory_per_patient,
        all_mandatory=sorted(all_vars),
        consistent_mandatory=sorted(consistent),
        mean_mandatory_count=round(mean_count, QAP.float_precision),
    )


def compare_active_passive(active_results: dict,
                           passive_results: dict,
                           n_total_variables: int = 10,
                           output_dir: str = None) -> ActiveVsPassiveResult:
    """Run the active vs passive comparison.

    Parameters
    ----------
    active_results : dict
        Phase 6 Valdez results.
    passive_results : dict
        C5 passive viewing results.
    n_total_variables : int
        Total number of candidate variables (for rate calculation).
    output_dir : str, optional
        Where to save results.
    """
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase6_valdez")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Summarise each dataset
    active_summary = summarise_dataset(
        active_results, "Valdez OSF nf7s8", "active")
    passive_summary = summarise_dataset(
        passive_results, "C5 Passive", "passive")

    # Jaccard comparisons
    active_all_set = set(active_summary.all_mandatory)
    passive_all_set = set(passive_summary.all_mandatory)
    active_consistent_set = set(active_summary.consistent_mandatory)
    passive_consistent_set = set(passive_summary.consistent_mandatory)

    j_all = jaccard(active_all_set, passive_all_set)
    j_consistent = jaccard(active_consistent_set, passive_consistent_set)

    unique_active = sorted(active_all_set - passive_all_set)
    unique_passive = sorted(passive_all_set - active_all_set)
    shared = sorted(active_all_set & passive_all_set)

    # Mandatory rates
    active_rate = (active_summary.mean_mandatory_count / n_total_variables
                   if n_total_variables > 0 else 0.0)
    passive_rate = (passive_summary.mean_mandatory_count / n_total_variables
                    if n_total_variables > 0 else 0.0)
    rate_diff = active_rate - passive_rate

    # Interpretation
    if j_consistent < QAP.jaccard_mandatory_threshold:
        interpretation = (
            "LOW OVERLAP (Jaccard={:.3f} < {:.2f}): Active and passive tasks "
            "produce substantially different mandatory variable sets. "
            "This supports the hypothesis that task engagement modulates "
            "which neural features are necessary for prediction.".format(
                j_consistent, QAP.jaccard_mandatory_threshold))
    else:
        interpretation = (
            "HIGH OVERLAP (Jaccard={:.3f} >= {:.2f}): Active and passive tasks "
            "produce similar mandatory variable sets. "
            "The core neural features needed for prediction may be "
            "task-invariant.".format(
                j_consistent, QAP.jaccard_mandatory_threshold))

    if unique_active:
        interpretation += (
            " Variables UNIQUE to active task: {}. "
            "These may reflect task-engagement-dependent neural mechanisms."
        ).format(", ".join(unique_active))

    governance_caveats = [
        "AI quality panel recommendation only — requires human validation",
        "C5 and Valdez use different patient cohorts — between-subject confounds",
        "Task difficulty, electrode placement, and patient demographics not matched",
    ]

    result = ActiveVsPassiveResult(
        timestamp=datetime.now().isoformat(),
        active_dataset="Valdez OSF nf7s8",
        passive_dataset="C5 Passive",
        active_summary=active_summary,
        passive_summary=passive_summary,
        jaccard_all=round(j_all, QAP.float_precision),
        jaccard_consistent=round(j_consistent, QAP.float_precision),
        unique_to_active=unique_active,
        unique_to_passive=unique_passive,
        shared=shared,
        active_mandatory_rate=round(active_rate, QAP.float_precision),
        passive_mandatory_rate=round(passive_rate, QAP.float_precision),
        rate_difference=round(rate_diff, QAP.float_precision),
        interpretation=interpretation,
        governance_caveats=governance_caveats,
    )

    # Save
    result_dict = asdict(result)
    with open(out_path / "active_vs_passive.json", 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # Print summary
    print_comparison(result)

    return result


def print_comparison(result: ActiveVsPassiveResult):
    """Print formatted comparison report."""
    print("\n" + "=" * 70)
    print("ACTIVE vs PASSIVE MANDATORY VARIABLE COMPARISON")
    print("=" * 70)

    print("\n--- Active: {} ({} patients) ---".format(
        result.active_dataset, result.active_summary.n_patients))
    print("  All mandatory:        {}".format(
        result.active_summary.all_mandatory))
    print("  Consistent mandatory: {}".format(
        result.active_summary.consistent_mandatory))
    print("  Mean count/patient:   {:.2f}".format(
        result.active_summary.mean_mandatory_count))

    print("\n--- Passive: {} ({} patients) ---".format(
        result.passive_dataset, result.passive_summary.n_patients))
    print("  All mandatory:        {}".format(
        result.passive_summary.all_mandatory))
    print("  Consistent mandatory: {}".format(
        result.passive_summary.consistent_mandatory))
    print("  Mean count/patient:   {:.2f}".format(
        result.passive_summary.mean_mandatory_count))

    print("\n--- Comparison ---")
    print("  Jaccard (all):        {:.4f}".format(result.jaccard_all))
    print("  Jaccard (consistent): {:.4f}".format(result.jaccard_consistent))
    print("  Shared:               {}".format(result.shared))
    print("  Unique to active:     {}".format(result.unique_to_active))
    print("  Unique to passive:    {}".format(result.unique_to_passive))
    print("  Rate active:          {:.4f}".format(result.active_mandatory_rate))
    print("  Rate passive:         {:.4f}".format(result.passive_mandatory_rate))
    print("  Rate difference:      {:.4f}".format(result.rate_difference))

    print("\n--- Interpretation ---")
    print("  {}".format(result.interpretation))

    print("\n--- Governance Caveats ---")
    for c in result.governance_caveats:
        print("  - {}".format(c))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_active_vs_passive(phase6_path: str = None,
                          c5_path: str = None) -> Optional[ActiveVsPassiveResult]:
    """Load both datasets and run comparison."""
    active = load_phase6_results(phase6_path)
    passive = load_c5_results(c5_path)

    if active is None:
        print("ERROR: Phase 6 (Valdez) results not found. "
              "Run run_p6_dual_arch first.")
        return None
    if passive is None:
        print("ERROR: C5 (passive) results not found. "
              "Run the C5 pipeline first.")
        return None

    return compare_active_passive(active, passive)


if __name__ == '__main__':
    run_active_vs_passive()
