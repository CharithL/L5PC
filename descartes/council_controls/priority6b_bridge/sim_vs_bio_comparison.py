"""
sim_vs_bio_comparison.py

Phase 6B-3: Compare C2 Simulation vs Real Data Mandatory Variables

Loads mandatory variable results from:
    - C2 simulation pipeline (L5PC biophysical model)
    - Phase 6B bridge pipeline (DANDI 000978 real hippocampal data)

Compares ONLY the shared variables (gamma_amp, theta_power, firing_rate,
synchrony) — sim-only variables (I_h, m_h, Ca_i) are excluded from the
bridge comparison because they cannot be measured in real recordings.

DECISION: Bridge SUPPORTED or FAILED
    SUPPORTED: >= bridge_overlap_required shared variables are mandatory
               in both sim and real data, AND Jaccard >= bridge_jaccard_min
    FAILED:    Otherwise

Usage:
    python -m descartes.council_controls.priority6b_bridge.sim_vs_bio_comparison
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
class VariableComparison:
    """Per-variable comparison between simulation and real data."""
    variable_name: str
    mandatory_in_sim: bool
    mandatory_in_real: bool
    sim_frequency: float   # fraction of seeds/sessions where mandatory
    real_frequency: float
    agreement: str         # "BOTH_MANDATORY", "SIM_ONLY", "REAL_ONLY", "NEITHER"


@dataclass
class BridgeVerdict:
    """Final bridge test decision."""
    timestamp: str
    # Sources
    sim_results_path: str
    real_results_path: str
    # Shared variables
    shared_variables: List[str]
    sim_only_variables: List[str]
    # Per-variable comparison
    variable_comparisons: List[VariableComparison]
    # Overlap
    both_mandatory: List[str]
    sim_only_mandatory: List[str]
    real_only_mandatory: List[str]
    neither_mandatory: List[str]
    # Metrics
    n_shared_overlap: int
    jaccard_shared: float
    # Decision
    verdict: str  # "SUPPORTED" or "FAILED"
    verdict_reason: str
    # Context
    interpretation: str
    implications_if_supported: str
    implications_if_failed: str
    governance_caveats: List[str]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sim_mandatory(results_path: str = None) -> Optional[Dict]:
    """Load C2 simulation mandatory variable results.

    Expected format:
    {
        "mandatory_variables": ["var1", "var2", ...],
        "variable_frequencies": {"var1": 0.95, "var2": 0.88, ...}
    }
    """
    if results_path is None:
        results_path = os.path.join(
            QAP.results_dir, "c2_simulation", "c2_mandatory.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


def load_real_mandatory(results_path: str = None) -> Optional[Dict]:
    """Load Phase 6B real data mandatory variable results."""
    if results_path is None:
        results_path = os.path.join(
            QAP.results_dir, "phase6b_bridge", "phase6b_results.json")
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def extract_mandatory_set(results: dict,
                          shared_only: bool = True) -> Set[str]:
    """Extract mandatory variable set from pipeline results.

    Handles both C2 simulation format and Phase 6B format.
    If shared_only=True, filters to QAP.bridge_shared_variables.
    """
    shared = set(QAP.bridge_shared_variables)

    # C2 format: direct list
    mandatory = set(results.get('mandatory_variables', []))

    # Phase 6B format: consistent_across_sessions
    if not mandatory:
        mandatory = set(results.get('consistent_across_sessions', []))

    # Fallback: all_session_consensus
    if not mandatory:
        mandatory = set(results.get('all_session_consensus', []))

    if shared_only:
        mandatory = mandatory & shared

    return mandatory


def extract_variable_frequencies(results: dict) -> Dict[str, float]:
    """Extract per-variable frequency from results."""
    # C2 format
    freqs = results.get('variable_frequencies', {})
    if freqs:
        return freqs

    # Phase 6B format: compute from session results
    session_results = results.get('session_results', [])
    if not session_results:
        return {}

    n_sessions = len(session_results)
    var_counts = {}
    for sr in session_results:
        for var in sr.get('consensus_mandatory', []):
            var_counts[var] = var_counts.get(var, 0) + 1

    return {var: count / n_sessions for var, count in var_counts.items()}


def compare_sim_vs_real(sim_results: dict,
                        real_results: dict,
                        output_dir: str = None) -> BridgeVerdict:
    """Run the full simulation vs real data comparison.

    Parameters
    ----------
    sim_results : dict
        C2 simulation mandatory variable results.
    real_results : dict
        Phase 6B real data results.
    output_dir : str, optional
        Where to save results.
    """
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase6b_bridge")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shared_vars = list(QAP.bridge_shared_variables)
    sim_only_vars = list(QAP.bridge_sim_only_variables)

    # Extract mandatory sets (shared variables only)
    sim_mandatory = extract_mandatory_set(sim_results, shared_only=True)
    real_mandatory = extract_mandatory_set(real_results, shared_only=True)

    # Frequencies
    sim_freqs = extract_variable_frequencies(sim_results)
    real_freqs = extract_variable_frequencies(real_results)

    # Per-variable comparison
    comparisons = []
    for var in shared_vars:
        in_sim = var in sim_mandatory
        in_real = var in real_mandatory
        if in_sim and in_real:
            agreement = "BOTH_MANDATORY"
        elif in_sim:
            agreement = "SIM_ONLY"
        elif in_real:
            agreement = "REAL_ONLY"
        else:
            agreement = "NEITHER"

        comparisons.append(VariableComparison(
            variable_name=var,
            mandatory_in_sim=in_sim,
            mandatory_in_real=in_real,
            sim_frequency=round(sim_freqs.get(var, 0.0), QAP.float_precision),
            real_frequency=round(real_freqs.get(var, 0.0), QAP.float_precision),
            agreement=agreement,
        ))

    # Overlap analysis
    both = sorted(sim_mandatory & real_mandatory)
    sim_only_mand = sorted(sim_mandatory - real_mandatory)
    real_only_mand = sorted(real_mandatory - sim_mandatory)
    neither = sorted(set(shared_vars) - sim_mandatory - real_mandatory)

    n_overlap = len(both)
    j = jaccard(sim_mandatory, real_mandatory)

    # --- DECISION ---
    min_overlap = QAP.bridge_overlap_required
    min_jaccard = QAP.bridge_jaccard_min

    if n_overlap >= min_overlap and j >= min_jaccard:
        verdict = "SUPPORTED"
        reason = (
            "Bridge test SUPPORTED: {} shared variables mandatory in both "
            "sim and real data (requires >= {}), Jaccard={:.3f} (requires "
            ">= {:.2f}).".format(n_overlap, min_overlap, j, min_jaccard))
    elif n_overlap >= min_overlap:
        verdict = "FAILED"
        reason = (
            "Bridge test FAILED: {} shared variables overlap (sufficient), "
            "but Jaccard={:.3f} < {:.2f} threshold. The mandatory sets "
            "diverge beyond acceptable limits.".format(
                n_overlap, j, min_jaccard))
    else:
        verdict = "FAILED"
        reason = (
            "Bridge test FAILED: only {} shared variables mandatory in both "
            "(requires >= {}). Simulation mandatory variables do not "
            "replicate in real recordings.".format(n_overlap, min_overlap))

    interpretation = (
        "The bridge test evaluates whether mandatory variables identified in "
        "biophysical simulation (C2, L5PC model) also emerge as mandatory "
        "when the same analysis pipeline is applied to real neural recordings. "
        "Only shared variables are compared — simulation-specific variables "
        "({}) cannot be measured in vivo and are excluded.".format(
            ", ".join(sim_only_vars)))

    implications_supported = (
        "If SUPPORTED: The simulation captures real neural dynamics "
        "sufficiently well that the same variables are necessary for "
        "prediction in both contexts. This strengthens confidence in "
        "simulation-derived insights for circuit C2, though it does NOT "
        "validate sim-only variables (I_h, m_h, Ca_i).")

    implications_failed = (
        "If FAILED: The simulation's mandatory variables do not replicate "
        "in real recordings. Possible explanations: (a) simulation lacks "
        "key biological mechanisms, (b) recording conditions differ too much, "
        "(c) species differences (rat vs model). Sim-derived mandatory "
        "variables should be treated with increased scepticism.")

    governance_caveats = [
        "AI quality panel recommendation only — requires human validation",
        "Bridge test uses rat data; human applicability requires separate validation",
        "Sim-only variables ({}) remain untested by this bridge".format(
            ", ".join(sim_only_vars)),
        "A SUPPORTED bridge does NOT validate the full simulation — only "
        "the shared variable subset",
        "Species, recording technology, and task differences are confounds",
    ]

    verdict_obj = BridgeVerdict(
        timestamp=datetime.now().isoformat(),
        sim_results_path=str(os.path.join(
            QAP.results_dir, "c2_simulation", "c2_mandatory.json")),
        real_results_path=str(os.path.join(
            QAP.results_dir, "phase6b_bridge", "phase6b_results.json")),
        shared_variables=shared_vars,
        sim_only_variables=sim_only_vars,
        variable_comparisons=comparisons,
        both_mandatory=both,
        sim_only_mandatory=sim_only_mand,
        real_only_mandatory=real_only_mand,
        neither_mandatory=neither,
        n_shared_overlap=n_overlap,
        jaccard_shared=round(j, QAP.float_precision),
        verdict=verdict,
        verdict_reason=reason,
        interpretation=interpretation,
        implications_if_supported=implications_supported,
        implications_if_failed=implications_failed,
        governance_caveats=governance_caveats,
    )

    # --- Save ---
    result_dict = asdict(verdict_obj)
    with open(out_path / "bridge_verdict.json", 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # --- Print ---
    print_verdict(verdict_obj)

    return verdict_obj


def print_verdict(v: BridgeVerdict):
    """Print formatted bridge verdict."""
    print("\n" + "=" * 70)
    print("SIMULATION-TO-BIOLOGY BRIDGE VERDICT")
    print("=" * 70)

    print("\n--- Variable Comparison (shared only) ---")
    print("{:<15} | {:>6} | {:>6} | {:>6} | {:>6} | {}".format(
        "Variable", "Sim?", "Real?", "SimFr", "RealFr", "Agreement"))
    print("-" * 70)
    for vc in v.variable_comparisons:
        print("{:<15} | {:>6} | {:>6} | {:>6.3f} | {:>6.3f} | {}".format(
            vc.variable_name,
            "YES" if vc.mandatory_in_sim else "no",
            "YES" if vc.mandatory_in_real else "no",
            vc.sim_frequency,
            vc.real_frequency,
            vc.agreement))

    print("\n--- Overlap ---")
    print("  Both mandatory:     {}".format(v.both_mandatory))
    print("  Sim-only mandatory: {}".format(v.sim_only_mandatory))
    print("  Real-only mandatory:{}".format(v.real_only_mandatory))
    print("  Neither:            {}".format(v.neither_mandatory))
    print("  Shared overlap:     {}".format(v.n_shared_overlap))
    print("  Jaccard (shared):   {:.4f}".format(v.jaccard_shared))

    print("\n--- VERDICT ---")
    print("  {}".format(v.verdict))
    print("  {}".format(v.verdict_reason))

    print("\n--- Governance Caveats ---")
    for c in v.governance_caveats:
        print("  - {}".format(c))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bridge_comparison(sim_path: str = None,
                          real_path: str = None) -> Optional[BridgeVerdict]:
    """Load both result sets and run comparison."""
    sim = load_sim_mandatory(sim_path)
    real = load_real_mandatory(real_path)

    if sim is None:
        print("ERROR: C2 simulation results not found. "
              "Run the C2 pipeline first.")
        return None
    if real is None:
        print("ERROR: Phase 6B real data results not found. "
              "Run run_p6b_bridge first.")
        return None

    return compare_sim_vs_real(sim, real)


if __name__ == '__main__':
    run_bridge_comparison()
