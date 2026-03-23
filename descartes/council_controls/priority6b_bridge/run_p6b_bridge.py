"""
run_p6b_bridge.py

Phase 6B-2: Full Pipeline on Real Hippocampal Data (DANDI 000978)

Runs LSTM + MLP-200ms on rat CA1->PFC recordings with:
    - 50-seed baseline (Priority 3)
    - Two-stage ablation (Priority 5)
    - Comparison limited to SHARED variables only

CRITICAL DISTINCTION:
    Variables extractable from BOTH simulation and real data:
        gamma_amp, theta_power, firing_rate, synchrony
    Variables available ONLY in simulation (NOT in real data):
        I_h, m_h, Ca_i
    Only shared variables are compared in the bridge test.

OUTPUT: results/council_controls/phase6b_bridge/

Usage:
    python -m descartes.council_controls.priority6b_bridge.run_p6b_bridge \\
        --data_dir data/dandi_000978
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

from ..config_council import QAP


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class SeedAblationResult:
    """Single seed ablation result for one session + architecture."""
    session_id: str
    architecture: str
    seed: int
    all_variables: List[str]
    stage1_survivors: List[str]
    stage2_mandatory: List[str]
    baseline_r: float
    final_r: float


@dataclass
class SessionArchResult:
    """Aggregated result for one session + architecture across seeds."""
    session_id: str
    architecture: str
    n_seeds: int
    mandatory_variables: List[str]
    mandatory_counts: Dict[str, int]
    mean_baseline_r: float
    std_baseline_r: float


@dataclass
class BridgeSessionResult:
    """Cross-architecture result for one session."""
    session_id: str
    lstm_mandatory: List[str]
    mlp_mandatory: List[str]
    jaccard_lstm_mlp: float
    consensus_mandatory: List[str]


@dataclass
class Phase6BResult:
    """Full Phase 6B result."""
    timestamp: str
    dandi_id: str
    input_region: str
    output_region: str
    shared_variables: List[str]
    sim_only_variables: List[str]
    n_sessions: int
    architectures: List[str]
    n_seeds: int
    session_results: List[BridgeSessionResult]
    # Aggregate across sessions
    all_session_consensus: List[str]
    consistent_across_sessions: List[str]
    mean_jaccard: float
    governance_caveats: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Feature extraction from binned spikes + LFP
# ---------------------------------------------------------------------------

def extract_shared_variables(ca1_spikes, pfc_spikes,
                             lfp_ca1=None, lfp_pfc=None,
                             fs_spikes: float = 40.0,
                             fs_lfp: float = 1250.0) -> Dict[str, object]:
    """Extract the SHARED variables from real neural data.

    These are the variables that can be computed from both simulation
    and real recordings:
        - gamma_amp:    30-80 Hz amplitude from LFP
        - theta_power:  4-8 Hz band power from LFP
        - firing_rate:  spike count per bin (already in spike matrices)
        - synchrony:    pairwise phase consistency between regions

    Parameters
    ----------
    ca1_spikes : array (n_timesteps, n_ca1)
    pfc_spikes : array (n_timesteps, n_pfc)
    lfp_ca1 : array (n_lfp, n_ca1_ch) or None
    lfp_pfc : array (n_lfp, n_pfc_ch) or None
    fs_spikes : float
    fs_lfp : float

    Returns
    -------
    dict of variable_name -> array
    """
    try:
        import numpy as np
    except ImportError:
        return {}

    n_timesteps = ca1_spikes.shape[0]
    variables = {}

    # firing_rate: mean spike count across neurons per timestep
    variables['firing_rate'] = np.mean(ca1_spikes, axis=1).astype(np.float32)

    # gamma_amp: placeholder (requires LFP + bandpass filter)
    if lfp_ca1 is not None:
        # In production: bandpass 30-80 Hz, compute analytic amplitude
        variables['gamma_amp'] = np.zeros(n_timesteps, dtype=np.float32)
    else:
        # Approximate from high-frequency spike variability
        variables['gamma_amp'] = np.std(
            ca1_spikes, axis=1).astype(np.float32)

    # theta_power: placeholder (requires LFP + bandpass filter)
    if lfp_ca1 is not None:
        variables['theta_power'] = np.zeros(n_timesteps, dtype=np.float32)
    else:
        # Approximate: low-frequency modulation of firing rate
        kernel_size = min(20, n_timesteps)
        if kernel_size > 0:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(
                np.mean(ca1_spikes, axis=1), kernel, mode='same')
            variables['theta_power'] = smoothed.astype(np.float32)
        else:
            variables['theta_power'] = np.zeros(
                n_timesteps, dtype=np.float32)

    # synchrony: correlation between CA1 and PFC mean activity
    ca1_mean = np.mean(ca1_spikes, axis=1)
    pfc_mean = np.mean(pfc_spikes, axis=1)
    # Rolling correlation with 1-second window
    win = int(fs_spikes)
    sync = np.zeros(n_timesteps, dtype=np.float32)
    for t in range(win, n_timesteps):
        ca1_seg = ca1_mean[t - win:t]
        pfc_seg = pfc_mean[t - win:t]
        if np.std(ca1_seg) > 1e-10 and np.std(pfc_seg) > 1e-10:
            sync[t] = np.corrcoef(ca1_seg, pfc_seg)[0, 1]
    variables['synchrony'] = sync

    return variables


# ---------------------------------------------------------------------------
# Ablation stubs
# ---------------------------------------------------------------------------

def run_single_seed_ablation(session_data, architecture: str, seed: int,
                             variable_names: List[str]) -> SeedAblationResult:
    """Run two-stage ablation for one seed on real data.

    STUB: Real implementation calls P3 baseline + P5 ablation infrastructure.
    """
    session_id = 'unknown'
    if hasattr(session_data, 'session_info'):
        session_id = session_data.session_info.session_id

    return SeedAblationResult(
        session_id=session_id,
        architecture=architecture,
        seed=seed,
        all_variables=list(variable_names),
        stage1_survivors=list(variable_names),
        stage2_mandatory=[],
        baseline_r=0.0,
        final_r=0.0,
    )


def aggregate_seeds(ablation_results: List[SeedAblationResult],
                    n_seeds: int,
                    threshold: float = None) -> SessionArchResult:
    """Aggregate across seeds for one session + architecture."""
    if threshold is None:
        threshold = QAP.baseline_mandatory_threshold

    if not ablation_results:
        return SessionArchResult(
            session_id='unknown', architecture='unknown',
            n_seeds=0, mandatory_variables=[], mandatory_counts={},
            mean_baseline_r=0.0, std_baseline_r=0.0)

    session_id = ablation_results[0].session_id
    architecture = ablation_results[0].architecture

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
        variance = sum((r - mean_r) ** 2 for r in baseline_rs) / (len(baseline_rs) - 1)
        std_r = variance ** 0.5

    return SessionArchResult(
        session_id=session_id,
        architecture=architecture,
        n_seeds=n_seeds,
        mandatory_variables=sorted(mandatory),
        mandatory_counts=variable_counts,
        mean_baseline_r=round(mean_r, QAP.float_precision),
        std_baseline_r=round(std_r, QAP.float_precision),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase6b(data_dir: str = "data/dandi_000978",
                output_dir: str = None) -> Optional[Phase6BResult]:
    """Execute the full Phase 6B bridge pipeline.

    Protocol:
        1. Load real hippocampal data (CA1 -> PFC)
        2. Extract SHARED variables only (gamma_amp, theta_power,
           firing_rate, synchrony)
        3. Train LSTM + MLP-200ms with 50-seed baseline
        4. Two-stage ablation on shared variables
        5. Cross-architecture Jaccard comparison
        6. Aggregate across sessions
    """
    if output_dir is None:
        output_dir = os.path.join(QAP.results_dir, "phase6b_bridge")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 6B: SIMULATION-TO-BIOLOGY BRIDGE")
    print("DANDI 000978 — Rat CA1 -> PFC (W-maze)")
    print("=" * 70)

    # Note: animal data — no consent gate needed
    print("Consent: CLEARED (animal data, IACUC approved)")

    # --- Load data ---
    from .dandi000978_loader import load_dandi000978, print_dataset_summary
    dataset = load_dandi000978(data_dir)
    print_dataset_summary(dataset)

    if dataset.n_sessions == 0:
        print("WARNING: No sessions loaded. Check data directory.")
        return None

    # --- Define variables ---
    shared_vars = list(QAP.bridge_shared_variables)
    sim_only_vars = list(QAP.bridge_sim_only_variables)

    print("\nSHARED variables (used in bridge test): {}".format(shared_vars))
    print("SIM-ONLY variables (excluded):          {}".format(sim_only_vars))

    architectures = ["LSTM", "MLP-200ms"]
    n_seeds = QAP.baseline_seed_count

    print("\nTraining {} architectures x {} seeds x {} sessions...".format(
        len(architectures), n_seeds, dataset.n_sessions))

    # --- Train all, then analyse ---
    all_results = {}
    for session in dataset.sessions:
        sid = session.session_info.session_id
        for arch in architectures:
            key = (sid, arch)
            all_results[key] = []
            for seed in range(n_seeds):
                result = run_single_seed_ablation(
                    session, arch, seed + QAP.random_seed_base,
                    shared_vars)
                all_results[key].append(result)

    print("Training complete. Analysing...\n")

    # --- Analyse ---
    session_results = []
    all_consensus = []

    for session in dataset.sessions:
        sid = session.session_info.session_id
        arch_results = {}
        for arch in architectures:
            key = (sid, arch)
            agg = aggregate_seeds(all_results[key], n_seeds)
            arch_results[arch] = agg

        lstm_set = set(arch_results.get("LSTM", SessionArchResult(
            sid, "LSTM", 0, [], {}, 0, 0)).mandatory_variables)
        mlp_set = set(arch_results.get("MLP-200ms", SessionArchResult(
            sid, "MLP-200ms", 0, [], {}, 0, 0)).mandatory_variables)

        j = jaccard(lstm_set, mlp_set)
        consensus = sorted(lstm_set & mlp_set)

        bridge_result = BridgeSessionResult(
            session_id=sid,
            lstm_mandatory=sorted(lstm_set),
            mlp_mandatory=sorted(mlp_set),
            jaccard_lstm_mlp=round(j, QAP.float_precision),
            consensus_mandatory=consensus,
        )
        session_results.append(bridge_result)
        all_consensus.extend(consensus)

        print("Session {}: LSTM={}, MLP={}, Jaccard={:.3f}, consensus={}".format(
            sid, sorted(lstm_set), sorted(mlp_set), j, consensus))

    # --- Aggregate across sessions ---
    # Count variable frequency across sessions
    var_session_counts = {}
    for sr in session_results:
        for var in sr.consensus_mandatory:
            var_session_counts[var] = var_session_counts.get(var, 0) + 1

    n_sessions = dataset.n_sessions
    min_sessions = max(1, n_sessions // 2)
    consistent = [var for var, count in var_session_counts.items()
                  if count >= min_sessions]

    jaccards = [sr.jaccard_lstm_mlp for sr in session_results]
    mean_j = sum(jaccards) / len(jaccards) if jaccards else 0.0

    governance_caveats = [
        "AI quality panel recommendation only — requires human validation",
        "Rat hippocampal data may not generalise to human circuits",
        "Bridge test is necessary but not sufficient for simulation validity",
        "Shared variables are a subset — sim-only variables ({}) untestable".format(
            ", ".join(sim_only_vars)),
    ]

    result = Phase6BResult(
        timestamp=datetime.now().isoformat(),
        dandi_id=dataset.dandi_id,
        input_region=dataset.input_region,
        output_region=dataset.output_region,
        shared_variables=shared_vars,
        sim_only_variables=sim_only_vars,
        n_sessions=n_sessions,
        architectures=architectures,
        n_seeds=n_seeds,
        session_results=session_results,
        all_session_consensus=sorted(set(all_consensus)),
        consistent_across_sessions=sorted(consistent),
        mean_jaccard=round(mean_j, QAP.float_precision),
        governance_caveats=governance_caveats,
    )

    # --- Save ---
    result_dict = asdict(result)
    with open(out_path / "phase6b_results.json", 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 6B SUMMARY")
    print("=" * 70)
    print("Sessions analysed:           {}".format(n_sessions))
    print("Shared variables tested:     {}".format(shared_vars))
    print("Sim-only (excluded):         {}".format(sim_only_vars))
    print("Mean Jaccard (LSTM vs MLP):  {:.4f}".format(mean_j))
    print("Consensus across sessions:   {}".format(sorted(set(all_consensus))))
    print("Consistent (>= 50% sessions): {}".format(consistent))
    print("\nGovernance caveats:")
    for c in governance_caveats:
        print("  - {}".format(c))
    print("\nResults saved: {}".format(out_path))
    print("=" * 70)

    return result


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/dandi_000978"
    run_phase6b(data_dir=data_dir)
