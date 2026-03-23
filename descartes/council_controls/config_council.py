"""
config_council.py

QAP Hyperparameters for DESCARTES Quality Controls Framework

NOTE: QAP = Quality-Assuring Panel = quality tool, NOT governance.
Genuine governance requires human critical friends and external review.

All constants used across Phases 0-7 are centralised here to ensure
reproducibility and prevent silent parameter drift between experiments.

Usage:
    from descartes.council_controls.config_council import QAP
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class _QAPConfig:
    """Immutable QAP hyperparameters across all phases."""

    # ---------------------------------------------------------------
    # Phase 0: Epistemic labeling
    # ---------------------------------------------------------------
    epistemic_downgrade_on_missing_control: bool = True

    # ---------------------------------------------------------------
    # Phase 1: Architecture control
    # ---------------------------------------------------------------
    mlp_window_sizes_ms: Tuple[int, ...] = (50, 100, 200, 500)
    mlp_hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    mlp_dropout: float = 0.3
    mlp_max_epochs: int = 300
    mlp_patience: int = 20
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 256
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_max_epochs: int = 300
    lstm_patience: int = 20

    # ---------------------------------------------------------------
    # Phase 2: Arbitrary probes (iAAFT surrogates)
    # ---------------------------------------------------------------
    iaaft_n_surrogates: int = 200
    iaaft_significance_alpha: float = 0.05
    iaaft_max_iterations: int = 100

    # ---------------------------------------------------------------
    # Phase 3: Baseline variance (50-seed ensemble)
    # ---------------------------------------------------------------
    baseline_seed_count: int = 50
    baseline_cv_method: str = "GroupKFold"
    baseline_groupkfold_n_splits: int = 5
    baseline_variance_report_percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95)
    # A variable is MANDATORY only if it survives ablation in >= this
    # fraction of the 50-seed baseline ensemble
    baseline_mandatory_threshold: float = 0.90

    # ---------------------------------------------------------------
    # Phase 4: Seizure / aura safety exclusion
    # ---------------------------------------------------------------
    seizure_exclusion_window_s: float = 300.0  # 5 minutes
    aura_exclusion_window_s: float = 600.0     # 10 minutes
    high_frequency_artifact_hz: float = 500.0

    # ---------------------------------------------------------------
    # Phase 5: Two-stage ablation
    # ---------------------------------------------------------------
    # Stage 1: coarse screen (drop each variable independently)
    ablation_stage1_metric: str = "pearson_r"
    ablation_stage1_threshold: float = 0.02
    # Stage 2: iterative removal from surviving set
    ablation_stage2_metric: str = "pearson_r"
    ablation_stage2_threshold: float = 0.02
    # Combined: Jaccard similarity between architectures
    jaccard_mandatory_threshold: float = 0.70

    # ---------------------------------------------------------------
    # Phase 6: Valdez active emotion (human iEEG)
    # ---------------------------------------------------------------
    valdez_osf_project: str = "nf7s8"
    valdez_trial_windows: Tuple[str, ...] = ("stimulus", "response")
    valdez_min_trials_per_patient: int = 10
    # Primary prediction: active emotion task variable mandatory in >= 50%
    valdez_mandatory_patient_fraction: float = 0.50
    # Architectures to run simultaneously
    valdez_architectures: Tuple[str, ...] = ("LSTM", "MLP-200ms", "PySR")

    # ---------------------------------------------------------------
    # Phase 6B: Simulation-to-biology bridge
    # ---------------------------------------------------------------
    dandi_000978_id: str = "000978"
    bridge_input_region: str = "CA1"
    bridge_output_region: str = "PFC"
    # Variables extractable from BOTH simulation and real data
    bridge_shared_variables: Tuple[str, ...] = (
        "gamma_amp", "theta_power", "firing_rate", "synchrony",
    )
    # Variables only available in simulation (NOT in real data)
    bridge_sim_only_variables: Tuple[str, ...] = (
        "I_h", "m_h", "Ca_i",
    )
    # Bridge verdict thresholds
    bridge_jaccard_min: float = 0.50
    bridge_overlap_required: int = 2  # minimum shared mandatory variables

    # ---------------------------------------------------------------
    # Phase 7: Monitoring protocol
    # ---------------------------------------------------------------
    monitoring_instruments: Tuple[str, ...] = (
        "MCQ",    # Metacognition Questionnaire
        "RDEES",  # Range and Differentiation of Emotional Experience Scale
        "EMA",    # Ecological Momentary Assessment
        "DES",    # Dissociative Experiences Scale
    )
    monitoring_control_phases: int = 3  # within-patient A-B-A design
    monitoring_override_policy: str = "patient_wins"
    monitoring_ppi_required: bool = True
    monitoring_microphenomenology: bool = True

    # ---------------------------------------------------------------
    # Cross-phase constants
    # ---------------------------------------------------------------
    random_seed_base: int = 42
    results_dir: str = "results/council_controls"
    float_precision: int = 6
    max_parallel_seeds: int = 10


# Singleton instance — import this
QAP = _QAPConfig()


def print_config():
    """Print all QAP hyperparameters for audit trail."""
    print("=" * 70)
    print("DESCARTES QAP CONFIGURATION")
    print("=" * 70)
    for f in QAP.__dataclass_fields__:
        val = getattr(QAP, f)
        print("  {:<45} = {}".format(f, val))
    print("=" * 70)


if __name__ == '__main__':
    print_config()
