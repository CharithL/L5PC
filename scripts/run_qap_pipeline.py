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


def run_phase1(args):
    """Phase 1: Architecture control (GPU)."""
    log.info("=" * 60)
    log.info("PHASE 1: ARCHITECTURE CONTROL")
    log.info("=" * 60)

    from descartes.council_controls.priority1_architecture.run_p1_control import (
        run_p1_control)

    circuit_data, targets, lstm_h_tr, lstm_h_un, lstm_pred, lstm_r2 = (
        load_circuit_data(args.processed_dir, args.subject, args.model_dir, args.device))

    result = run_p1_control(
        circuit_data=circuit_data,
        targets=targets,
        results_dir='results/council_controls',
        dt_ms=10.0,
        device=args.device,
        include_lstm=(lstm_h_tr is not None),
        lstm_hidden_trained=lstm_h_tr,
        lstm_hidden_untrained=lstm_h_un,
        lstm_output_pred=lstm_pred,
        lstm_output_r2=lstm_r2,
    )

    log.info("Phase 1 decision: %s", result.get('decision', 'UNKNOWN'))
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
