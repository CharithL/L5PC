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


def _import_phase34():
    """Import the original Phase 3-4 module by file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kyzar_phase3_4",
        Path(__file__).parent / "run_kyzar_phase3_4.py")
    phase34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase34)
    return phase34


def _filter_dead_neurons(H_trained, H_untrained, threshold=1e-10):
    """Filter out zero-variance columns from BOTH matrices using same mask.

    Bug C fix: trained MLPs have ~50% dead ReLU neurons. Remove them from
    BOTH trained and untrained to ensure symmetric feature spaces.
    """
    var_trained = np.std(H_trained, axis=0)
    var_untrained = np.std(H_untrained, axis=0)
    alive = (var_trained > threshold) | (var_untrained > threshold)
    n_dead = int((~alive).sum())
    n_total = len(alive)
    if n_dead > 0:
        log.info("    Filtered %d/%d dead neurons (%.0f%% alive)",
                 n_dead, n_total, 100 * alive.sum() / n_total)
    return H_trained[:, alive], H_untrained[:, alive], alive


def _extract_mandatory(phase4_results):
    """Extract mandatory variable names from Phase 4 results.

    Bug A fix: Phase 4 nests verdict under 'resample_ablation' key.
    """
    mandatory = []
    for vname, vresult in phase4_results.items():
        if not isinstance(vresult, dict):
            continue
        # Phase 4 returns {vname: {'resample_ablation': {...}, 'epoch_ablation': {...}}}
        abl = vresult.get('resample_ablation', vresult)
        if abl.get('overall_verdict') == 'MANDATORY':
            mandatory.append(vname)
    return mandatory


def _probe_architecture_with_original_pipeline(arch_name, H_trained, H_untrained,
                                                 session_data, phase34_module):
    """Probe one architecture using the EXACT same Phase 3-4 functions.

    Bug B fix: ONE probing code path for ALL architectures. Creates a
    synthetic session_data dict with the MLP's hidden states but the
    original bio targets, trial groups, and epoch masks.
    """
    # Filter dead neurons (Bug C)
    H_tr_filt, H_un_filt, alive_mask = _filter_dead_neurons(H_trained, H_untrained)

    log.info("  === Architecture: %s ===", arch_name)
    log.info("    H_trained: %s -> %s after filtering",
             H_trained.shape, H_tr_filt.shape)
    log.info("    H_untrained: %s -> %s",
             H_untrained.shape, H_un_filt.shape)

    # Build synthetic session_data with MLP hidden states but original bio data
    n_mlp = H_tr_filt.shape[0]
    n_orig = session_data['bio_targets'].shape[0]
    n = min(n_mlp, n_orig)

    arch_session = {
        'H_trained': H_tr_filt[:n].astype(np.float64),
        'H_untrained': H_un_filt[:n].astype(np.float64),
        'bio_targets': session_data['bio_targets'][:n],
        'bio_names': session_data['bio_names'],
        'epoch_mask': session_data['epoch_mask'][:n],
        'trial_groups': session_data['trial_groups'][:n],
        'meta': session_data['meta'],
        'model_info': session_data['model_info'],
        'hidden_dim': H_tr_filt.shape[1],
    }

    log.info("    Aligned samples: %d (MLP=%d, original=%d)", n, n_mlp, n_orig)

    # Call the EXACT same Phase 3 probing
    p3 = phase34_module.run_phase3(arch_session)

    # Call the EXACT same Phase 4 ablation
    p4 = phase34_module.run_phase4(arch_session, p3)

    mandatory = _extract_mandatory(p4)
    log.info("    MANDATORY: %s",
             ', '.join(mandatory) if mandatory else 'NONE')

    return {
        'architecture': arch_name,
        'phase3': p3,
        'phase4': p4,
        'mandatory': mandatory,
        'n_samples': n,
        'hidden_dim_raw': H_trained.shape[1],
        'hidden_dim_filtered': H_tr_filt.shape[1],
        'n_dead_neurons': int((~alive_mask).sum()),
    }


def run_phase1(args):
    """Phase 1: Architecture control (GPU).

    ONE probing function for ALL architectures — imported from run_kyzar_phase3_4.py.
    The ONLY difference between LSTM and MLP is the hidden state matrix.
    """
    log.info("=" * 60)
    log.info("PHASE 1: ARCHITECTURE CONTROL")
    log.info("  ONE probing path for all architectures")
    log.info("=" * 60)

    phase34 = _import_phase34()

    # Step 1: Load original session data + run LSTM baseline
    log.info("\n--- STEP 1: LSTM BASELINE (original Phase 3-4) ---")
    session_data = phase34.load_session(
        args.processed_dir, args.model_dir, args.subject, hidden_dim=64)

    if session_data is None:
        log.error("Cannot load session data for sub-%s", args.subject)
        return {'decision': 'ERROR', 'reasoning': 'Session data not found'}

    log.info("  Loaded sub-%s: H=%s, %d bio vars, %d trials, CC=%.3f",
             args.subject, session_data['H_trained'].shape,
             len(session_data['bio_names']),
             len(np.unique(session_data['trial_groups'])),
             session_data['model_info'].get('output_cc', 0))

    # Run original Phase 3+4 on LSTM
    log.info("  Running Phase 3 probing on LSTM...")
    lstm_p3 = phase34.run_phase3(session_data)
    log.info("  Running Phase 4 ablation on LSTM...")
    lstm_p4 = phase34.run_phase4(session_data, lstm_p3)
    lstm_mandatory = _extract_mandatory(lstm_p4)

    log.info("  LSTM MANDATORY: %s",
             ', '.join(lstm_mandatory) if lstm_mandatory else 'NONE')

    if not lstm_mandatory:
        log.warning("  WARNING: LSTM baseline found no mandatory variables.")
        log.warning("  This subject may be a zombie, or Phase 2 needs re-running.")

    # Step 2: Train + probe MLP architectures using SAME probing pipeline
    log.info("\n--- STEP 2: MLP ARCHITECTURES ---")

    # Load raw X, Y for MLP training
    sess_name = 'session_sub{}_ses2'.format(args.subject)
    data_dir = Path(args.processed_dir) / sess_name
    X_data = dict(np.load(data_dir / 'X_trials.npz'))
    Y_data = dict(np.load(data_dir / 'Y_trials.npz'))

    meta = session_data['meta']
    X_list, Y_list = [], []
    for ti in range(meta['n_trials']):
        key = 'trial_{}'.format(ti)
        if key in X_data and key in Y_data:
            X_list.append(X_data[key])
            Y_list.append(Y_data[key])

    X_seq = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_seq = np.concatenate(Y_list, axis=0).astype(np.float32)

    from descartes.council_controls.priority1_architecture.run_p1_control import (
        _train_mlp_architecture)

    MLP_WINDOWS = [50, 100, 200, 500]
    all_arch_results = []

    # LSTM result
    all_arch_results.append({
        'architecture': 'LSTM',
        'mandatory': lstm_mandatory,
        'phase3': lstm_p3,
        'phase4': lstm_p4,
        'n_samples': session_data['H_trained'].shape[0],
        'hidden_dim_raw': session_data['H_trained'].shape[1],
        'hidden_dim_filtered': session_data['H_trained'].shape[1],
        'n_dead_neurons': 0,
    })

    for window_ms in MLP_WINDOWS:
        log.info("\n  Training MLP-%dms...", window_ms)
        try:
            H_tr, H_un, out_pred, test_r2, w_bins = _train_mlp_architecture(
                X_seq, Y_seq, window_ms, dt_ms=10.0, device=args.device)

            # CRITICAL: align to match original bio targets
            # MLP hidden state[i] corresponds to input timestep (i + w_bins - 1)
            # Original bio targets start at timestep 0
            # Slice bio targets to start at (w_bins - 1) to align
            offset = w_bins - 1
            log.info("    Output R2: %.3f, offset: %d timesteps", test_r2, offset)

            # Create offset session_data
            n_mlp = H_tr.shape[0]
            n_orig = session_data['bio_targets'].shape[0]
            n = min(n_mlp, n_orig - offset)

            if n < 200:
                log.warning("    Too few aligned samples (%d) — skipping", n)
                continue

            offset_session = {
                'H_trained': session_data['H_trained'],  # placeholder, replaced below
                'H_untrained': session_data['H_untrained'],
                'bio_targets': session_data['bio_targets'][offset:offset + n],
                'bio_names': session_data['bio_names'],
                'epoch_mask': session_data['epoch_mask'][offset:offset + n],
                'trial_groups': session_data['trial_groups'][offset:offset + n],
                'meta': session_data['meta'],
                'model_info': {'output_cc': float(np.sqrt(max(test_r2, 0)))},
                'hidden_dim': H_tr.shape[1],
            }

            result = _probe_architecture_with_original_pipeline(
                'MLP_{}ms'.format(window_ms),
                H_tr[:n], H_un[:n],
                offset_session, phase34)
            result['output_r2'] = test_r2
            result['window_offset'] = offset
            all_arch_results.append(result)

        except Exception as e:
            log.error("    MLP-%dms failed: %s", window_ms, e)
            import traceback
            traceback.print_exc()
            continue

    # Step 3: Build comparison table
    log.info("\n--- STEP 3: ARCHITECTURE COMPARISON ---")
    log.info("%-15s | %-8s | %-6s | Mandatory Variables", "Architecture", "Samples", "H_dim")
    log.info("-" * 70)
    for r in all_arch_results:
        mvars = ', '.join(r['mandatory']) if r['mandatory'] else 'NONE'
        log.info("%-15s | %-8d | %-6d | %s",
                 r['architecture'], r['n_samples'],
                 r['hidden_dim_filtered'], mvars)

    # Decision logic
    non_lstm = [r for r in all_arch_results if r['architecture'] != 'LSTM']
    all_mandatory = set()
    for r in all_arch_results:
        all_mandatory.update(r['mandatory'])

    architecture_invariant = []
    lstm_specific = []
    for vname in all_mandatory:
        in_lstm = vname in lstm_mandatory
        n_non_lstm = sum(1 for r in non_lstm if vname in r['mandatory'])
        if n_non_lstm >= 2:
            architecture_invariant.append(vname)
        elif in_lstm and n_non_lstm == 0:
            lstm_specific.append(vname)

    if architecture_invariant:
        decision = 'PASS'
        reasoning = ('{} variable(s) architecture-invariant: {}. '
                     'Genuine biophysical encoding confirmed.'.format(
                         len(architecture_invariant), architecture_invariant))
    elif lstm_specific:
        decision = 'FAIL'
        reasoning = ('All mandatory variables are LSTM-specific: {}. '
                     'May be architecture artifacts.'.format(lstm_specific))
    else:
        decision = 'INCONCLUSIVE'
        reasoning = ('No mandatory variables found in any architecture, '
                     'or insufficient non-LSTM architectures tested.')

    log.info("\n" + "=" * 60)
    log.info("P1 DECISION: %s", decision)
    log.info("  Architecture-invariant: %s", architecture_invariant)
    log.info("  LSTM-specific: %s", lstm_specific)
    log.info("  Reasoning: %s", reasoning)
    log.info("=" * 60)

    # Save results
    output = {
        'decision': decision,
        'reasoning': reasoning,
        'architecture_invariant': architecture_invariant,
        'lstm_specific': lstm_specific,
        'subject': args.subject,
        'per_architecture': [{
            'architecture': r['architecture'],
            'mandatory': r['mandatory'],
            'n_samples': r['n_samples'],
            'hidden_dim_raw': r.get('hidden_dim_raw', 0),
            'hidden_dim_filtered': r.get('hidden_dim_filtered', 0),
            'n_dead_neurons': r.get('n_dead_neurons', 0),
        } for r in all_arch_results],
    }

    results_dir = Path('results/council_controls')
    results_dir.mkdir(parents=True, exist_ok=True)

    import json as _json

    class _NpEnc(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_dir / 'p1_architecture_control.json', 'w') as f:
        _json.dump(output, f, indent=2, cls=_NpEnc)

    return output


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
