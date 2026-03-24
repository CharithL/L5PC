#!/usr/bin/env python3
"""
run_qap_pipeline.py

CLI runner for DESCARTES Quality-Assured Controls (QAP) pipeline.
Executes phases sequentially per the guide's dependency map.

Usage:
  # Day One (CPU): Phase 0 governance
  python scripts/run_qap_pipeline.py --phase 0

  # Phase 1: Architecture control (GPU, needs circuit data)
  python scripts/run_qap_pipeline.py --phase 1 \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --circuit C6 --subject 5 --device cuda

  # Phase 3: 50-seed baseline variance
  python scripts/run_qap_pipeline.py --phase 3 \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --circuit C6 --subject 5 --device cuda

  # Phase 5: Two-stage ablation
  python scripts/run_qap_pipeline.py --phase 5 \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --circuit C6 --subject 5 --device cuda

  # All GPU phases sequentially
  python scripts/run_qap_pipeline.py --phase all \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --circuit C6 --subject 5 --device cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('qap_pipeline')


def load_circuit_data(processed_dir, subject, model_dir=None, device='cpu'):
    """Load circuit data for a Kyzar subject into the format P1 expects."""
    import torch
    from scipy.signal import butter, filtfilt, hilbert

    sess = 'session_sub{}_ses2'.format(subject)
    sess_dir = Path(processed_dir) / sess

    X_data = dict(np.load(sess_dir / 'X_trials.npz'))
    Y_data = dict(np.load(sess_dir / 'Y_trials.npz'))

    with open(sess_dir / 'metadata.json') as f:
        meta = json.load(f)

    # Concatenate all trials
    X_list, Y_list = [], []
    for ti in range(meta['n_trials']):
        key = 'trial_{}'.format(ti)
        if key in X_data and key in Y_data:
            X_list.append(X_data[key])
            Y_list.append(Y_data[key])

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    Y = np.concatenate(Y_list, axis=0).astype(np.float32)

    trial_lengths = [x.shape[0] for x in X_list]
    trial_size = int(np.mean(trial_lengths))

    # Compute bio targets
    fs = 100.0  # 10ms bins
    pop_rate = np.mean(X, axis=1)
    T = len(pop_rate)

    targets = {}
    targets['firing_rate_input'] = pop_rate

    # Theta power
    try:
        nyq = fs / 2
        b, a = butter(3, [4/nyq, 8/nyq], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        filt = filtfilt(b, a, pop_rate, padlen=padlen)
        targets['theta_power'] = np.abs(hilbert(filt)) ** 2
    except Exception:
        targets['theta_power'] = np.zeros(T)

    # Gamma power
    try:
        gmax = min(80, fs/2 - 1)
        b, a = butter(3, [30/nyq, gmax/nyq], btype='band')
        filt = filtfilt(b, a, pop_rate, padlen=padlen)
        targets['gamma_power'] = np.abs(hilbert(filt)) ** 2
    except Exception:
        targets['gamma_power'] = np.zeros(T)

    # Theta-gamma PAC
    try:
        theta_phase = np.angle(hilbert(filtfilt(
            *butter(3, [4/nyq, 8/nyq], btype='band'), pop_rate, padlen=padlen)))
        gamma_amp = np.sqrt(np.maximum(targets['gamma_power'], 0))
        win = 50
        half = win // 2
        pac = np.zeros(T)
        for i in range(T):
            lo, hi = max(0, i - half), min(T, i + half + 1)
            if hi - lo < 5:
                continue
            z = np.mean(gamma_amp[lo:hi] * np.exp(1j * theta_phase[lo:hi]))
            pac[i] = np.abs(z)
        targets['theta_gamma_pac'] = pac
    except Exception:
        targets['theta_gamma_pac'] = np.zeros(T)

    # Trial variance
    win = 20
    mean_fr = np.mean(np.concatenate([X, Y], axis=1), axis=1)
    tv = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        tv[i] = np.var(mean_fr[lo:hi]) if hi - lo > 1 else 0
    targets['trial_variance'] = tv

    # Load LSTM hidden states if available
    lstm_h_tr, lstm_h_un, lstm_pred, lstm_r2 = None, None, None, 0.0
    if model_dir:
        model_sess = Path(model_dir) / sess
        for hdir in sorted(model_sess.glob('lstm_h*')):
            h_path = hdir / 'hidden_trained.npz'
            hu_path = hdir / 'hidden_untrained.npz'
            val_path = hdir / 'output_validation.json'
            if h_path.exists():
                h_dict = dict(np.load(h_path))
                h_arrays = [h_dict[k] for k in sorted(h_dict.keys()) if k.startswith('trial_')]
                lstm_h_tr = np.concatenate(h_arrays, axis=0)[:T]

            if hu_path.exists():
                hu_dict = dict(np.load(hu_path))
                hu_arrays = [hu_dict[k] for k in sorted(hu_dict.keys()) if k.startswith('trial_')]
                lstm_h_un = np.concatenate(hu_arrays, axis=0)[:T]

            if val_path.exists():
                with open(val_path) as f:
                    val = json.load(f)
                lstm_r2 = val.get('output_cc', 0.0) ** 2
                lstm_pred = np.zeros((T, Y.shape[1]))  # placeholder

    circuit_data = {
        'C6_sub{}'.format(subject): {
            'X': X, 'Y': Y, 'trial_size': trial_size,
        }
    }

    return circuit_data, targets, lstm_h_tr, lstm_h_un, lstm_pred, lstm_r2


def run_phase0():
    """Phase 0 + 0B: Governance (CPU only)."""
    log.info("=" * 60)
    log.info("PHASE 0: GOVERNANCE")
    log.info("=" * 60)

    # Epistemic labels (library module — no standalone runner, just import check)
    from descartes.council_controls.priority0_epistemic.epistemic_labels import (
        label_result, EpistemicPhase, EpistemicStatus)
    log.info("  Epistemic labeling module loaded OK")

    # Consent audit
    from descartes.council_controls.priority0b_governance.consent_audit import run_audit
    run_audit()

    # DURA matrix
    from descartes.council_controls.priority0b_governance.dura_matrix import generate_dura
    generate_dura()

    # Governance checklist (library module — no standalone runner)
    from descartes.council_controls.priority0b_governance.governance_checklist import (
        check_governance)
    log.info("  Governance checklist module loaded OK")

    log.info("Phase 0 complete. Review results/governance/ before proceeding.")


def validate_lstm_baseline_original(processed_dir, model_dir, subject_id, hidden_dim=64):
    """Bug 1 fix: Run the EXACT original Phase 3-4 probing on this subject.

    Uses the same load_session, ridge_delta_r2, and resample_ablation functions
    from run_kyzar_phase3_4.py — NOT a reimplementation.

    Returns the original session_data dict and phase3 results for use in P1.
    """
    # Import the EXACT functions from the original Phase 3-4 script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kyzar_phase3_4",
        Path(__file__).parent / "run_kyzar_phase3_4.py")
    phase34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase34)

    log.info("=== LSTM BASELINE VALIDATION (sub-%s) ===", subject_id)
    log.info("  Using EXACT original Phase 3-4 code path")

    # Load using original loader (gets pre-computed bio_targets.npz, proper trial groups)
    session_data = phase34.load_session(processed_dir, model_dir, subject_id, hidden_dim)
    if session_data is None:
        log.error("  FAILED: Could not load session data for sub-%s", subject_id)
        log.error("  Check: processed_dir=%s, model_dir=%s, h=%d",
                  processed_dir, model_dir, hidden_dim)
        return None, None

    log.info("  H_trained: %s", session_data['H_trained'].shape)
    log.info("  H_untrained: %s", session_data['H_untrained'].shape)
    log.info("  Bio targets: %s (%d variables)",
             session_data['bio_targets'].shape, len(session_data['bio_names']))
    log.info("  Trial groups: %d unique", len(np.unique(session_data['trial_groups'])))
    log.info("  CC: %.3f", session_data['model_info'].get('output_cc', 0))

    # Run original Phase 3 probing
    log.info("  Running original Phase 3 probing (all %d variables)...",
             len(session_data['bio_names']))
    phase3_results = phase34.run_phase3(session_data)

    # Check theta_gamma_pac
    pac_result = phase3_results.get('theta_gamma_pac', {})
    pac_ridge = pac_result.get('ridge', {})
    pac_dr2 = pac_ridge.get('delta_r2', 0)
    log.info("  theta_gamma_pac: dR2=%.4f (trained=%.4f, untrained=%.4f)",
             pac_dr2, pac_ridge.get('r2_trained', 0), pac_ridge.get('r2_untrained', 0))

    # Run Phase 4 ablation on candidates
    log.info("  Running original Phase 4 ablation...")
    phase4_results = phase34.run_phase4(session_data, phase3_results)

    # Check theta_gamma_pac ablation
    pac_abl = phase4_results.get('theta_gamma_pac', {})
    if pac_abl:
        log.info("  theta_gamma_pac ablation: %s", pac_abl.get('overall_verdict', 'N/A'))
        for k, v in pac_abl.get('per_k', {}).items():
            log.info("    k=%s: z=%.2f %s", k, v.get('z_score', 0), v.get('verdict', ''))

    # Summary of mandatory variables
    mandatory = []
    for vname, vresult in phase4_results.items():
        if isinstance(vresult, dict) and vresult.get('overall_verdict') == 'MANDATORY':
            mandatory.append(vname)
    log.info("  MANDATORY variables: %s",
             ', '.join(mandatory) if mandatory else 'NONE')

    if not mandatory:
        log.warning("  WARNING: No mandatory variables found by original pipeline.")
        log.warning("  If this subject was previously non-zombie, something has changed.")
    else:
        log.info("  LSTM baseline VALIDATED — %d mandatory variables found", len(mandatory))

    log.info("=== LSTM BASELINE VALIDATION COMPLETE ===")

    return session_data, phase3_results, phase4_results


def run_phase1(args):
    """Phase 1: Architecture control (GPU).

    Step 1: Validate LSTM baseline using the EXACT original Phase 3-4 code.
    Step 2: Run MLP/PySR architecture comparison using the original bio targets
            and the original probing functions.
    """
    log.info("=" * 60)
    log.info("PHASE 1: ARCHITECTURE CONTROL")
    log.info("=" * 60)

    # Step 1: Run original Phase 3-4 on this subject as the LSTM baseline
    validation = validate_lstm_baseline_original(
        args.processed_dir, args.model_dir, args.subject)

    if validation is None:
        log.error("LSTM baseline validation failed — cannot proceed with P1")
        return {'decision': 'ERROR', 'reasoning': 'LSTM baseline validation failed'}

    session_data, phase3_results, phase4_results = validation

    # Extract the original bio targets and probing data for MLP comparison
    # The LSTM results are already computed by the original pipeline above.
    # Now we need to run MLP architectures using the SAME targets and groups.
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        run_p1_control)

    # Build circuit_data from the original session
    sess_name = 'session_sub{}_ses2'.format(args.subject)
    data_dir = Path(args.processed_dir) / sess_name
    X_data = dict(np.load(data_dir / 'X_trials.npz'))
    Y_data = dict(np.load(data_dir / 'Y_trials.npz'))

    # Concatenate all trials (same order as original)
    meta = session_data['meta']
    X_list, Y_list = [], []
    for ti in range(meta['n_trials']):
        key = 'trial_{}'.format(ti)
        if key in X_data and key in Y_data:
            X_list.append(X_data[key])
            Y_list.append(Y_data[key])

    X_seq = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_seq = np.concatenate(Y_list, axis=0).astype(np.float32)

    # Use the ORIGINAL bio targets (from bio_targets.npz, not recomputed)
    bio_names = session_data['bio_names']
    bio_targets_mat = session_data['bio_targets']  # (T, 18)
    original_targets = {}
    for vi, vname in enumerate(bio_names):
        original_targets[vname] = bio_targets_mat[:, vi]

    circuit_data = {
        'C6_sub{}'.format(args.subject): {
            'X': X_seq, 'Y': Y_seq,
            'trial_size': int(np.mean([x.shape[0] for x in X_list])),
        }
    }

    # Step 2: Run P1 architecture comparison with ORIGINAL targets
    # Include LSTM hidden states from original pipeline
    result = run_p1_control(
        circuit_data=circuit_data,
        targets=original_targets,
        results_dir='results/council_controls',
        dt_ms=10.0,
        device=args.device,
        include_lstm=True,
        lstm_hidden_trained=session_data['H_trained'],
        lstm_hidden_untrained=session_data['H_untrained'],
        lstm_output_pred=np.zeros((len(session_data['H_trained']), Y_seq.shape[1])),
        lstm_output_r2=session_data['model_info'].get('output_cc', 0) ** 2,
    )

    # Attach original Phase 3-4 results for comparison
    result['original_phase3'] = {vname: {
        'ridge_dr2': r.get('ridge', {}).get('delta_r2', 0),
        'mlp_dr2': r.get('mlp', {}).get('delta_r2', 0),
    } for vname, r in phase3_results.items()}
    result['original_phase4_mandatory'] = [
        vname for vname, r in phase4_results.items()
        if isinstance(r, dict) and r.get('overall_verdict') == 'MANDATORY'
    ]

    log.info("Phase 1 decision: %s", result.get('decision', 'UNKNOWN'))
    log.info("Original Phase 3-4 mandatory: %s", result['original_phase4_mandatory'])
    return result


def run_phase3(args):
    """Phase 3: 50-seed baseline variance."""
    log.info("=" * 60)
    log.info("PHASE 3: 50-SEED BASELINE VARIANCE")
    log.info("=" * 60)

    from descartes.council_controls.priority3_baseline_variance.run_50_seeds import (
        run_seed_sweep)
    from descartes.council_controls.priority3_baseline_variance.baseline_distribution import (
        analyze_distributions)
    from descartes.council_controls.priority3_baseline_variance.retrofit_all_circuits import (
        run_retrofit)

    log.info("Running 50 untrained seeds...")
    run_seed_sweep()

    log.info("Computing baseline distribution...")
    analyze_distributions()

    log.info("Retrofitting trained results against baseline...")
    run_retrofit()


def run_phase5(args):
    """Phase 5: Two-stage ablation."""
    log.info("=" * 60)
    log.info("PHASE 5: TWO-STAGE ABLATION")
    log.info("=" * 60)

    from descartes.council_controls.priority5_twostage_ablation.correlation_structure import (
        analyze_correlation_structure)
    from descartes.council_controls.priority5_twostage_ablation.twostage_classify import (
        run_twostage_classification)

    _, targets, lstm_h_tr, _, _, _ = (
        load_circuit_data(args.processed_dir, args.subject, args.model_dir, args.device))

    if lstm_h_tr is None:
        log.error("No LSTM hidden states found. Run Phase 2 first.")
        return

    results_dir = 'results/council_controls/phase5_twostage'

    log.info("Computing correlation structure...")
    analyze_correlation_structure(
        hidden_states=lstm_h_tr, output_dir=results_dir)

    # Two-stage classification needs a probe target — use first bio target
    first_target_name = list(targets.keys())[0]
    first_target = targets[first_target_name]
    n = min(len(first_target), lstm_h_tr.shape[0])

    log.info("Running two-stage classification on %s...", first_target_name)
    run_twostage_classification(
        hidden_states=lstm_h_tr[:n],
        probe_target=first_target[:n],
        output_dir=results_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES QAP Pipeline Runner')
    parser.add_argument('--phase', required=True,
                        help='Phase to run: 0, 1, 3, 5, or all')
    parser.add_argument('--processed-dir',
                        help='Path to kyzar_processed directory')
    parser.add_argument('--model-dir',
                        help='Path to models/kyzar directory')
    parser.add_argument('--circuit', default='C6',
                        help='Circuit (default: C6)')
    parser.add_argument('--subject', default='5',
                        help='Subject number (default: 5)')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu or cuda)')
    args = parser.parse_args()

    t_start = time.time()

    if args.phase in ('0', 'all'):
        run_phase0()

    if args.phase in ('1', 'all'):
        if not args.processed_dir:
            log.error("--processed-dir required for Phase 1")
            sys.exit(1)
        run_phase1(args)

    if args.phase in ('3', 'all'):
        if not args.processed_dir:
            log.error("--processed-dir required for Phase 3")
            sys.exit(1)
        run_phase3(args)

    if args.phase in ('5', 'all'):
        if not args.processed_dir:
            log.error("--processed-dir required for Phase 5")
            sys.exit(1)
        run_phase5(args)

    total = time.time() - t_start
    log.info("\nQAP Pipeline phase=%s complete (%.1f min)", args.phase, total / 60)


if __name__ == '__main__':
    main()
