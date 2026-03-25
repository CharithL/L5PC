#!/usr/bin/env python3
"""Quick test: run P1 MLP-50ms only on sub-2 to verify intersection filter fix."""

import sys
import os
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('test_mlp50')

KYZAR_DIR = r"C:\Users\chari\OneDrive\Documents\Descartes_Cogito\Kyzar\data\kyzar_processed"
MODEL_DIR = r"C:\Users\chari\OneDrive\Documents\Descartes_Cogito\L5PC\models\kyzar"
SUBJECT = '2'


def main():
    log.info("=" * 60)
    log.info("QUICK TEST: MLP-50ms on sub-%s with intersection filter", SUBJECT)
    log.info("=" * 60)

    # Import Phase 3-4 module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kyzar_phase3_4",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_kyzar_phase3_4.py"))
    phase34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase34)

    # Import MLP trainer from P1
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        _train_mlp_architecture)

    # Load session data
    log.info("Loading session data...")
    session_data = phase34.load_session(KYZAR_DIR, MODEL_DIR, SUBJECT, hidden_dim=64)
    if session_data is None:
        log.error("Failed to load session data")
        return 1

    log.info("  H_trained: %s, bio_vars: %d, trials: %d",
             session_data['H_trained'].shape,
             len(session_data['bio_names']),
             len(np.unique(session_data['trial_groups'])))

    # Load raw X, Y for MLP training
    sess_name = 'session_sub{}_ses2'.format(SUBJECT)
    data_dir = os.path.join(KYZAR_DIR, sess_name)
    X_data = dict(np.load(os.path.join(data_dir, 'X_trials.npz')))
    Y_data = dict(np.load(os.path.join(data_dir, 'Y_trials.npz')))
    meta = session_data['meta']

    X_list, Y_list = [], []
    for ti in range(meta['n_trials']):
        key = 'trial_{}'.format(ti)
        if key in X_data and key in Y_data:
            X_list.append(X_data[key])
            Y_list.append(Y_data[key])

    X_seq = np.concatenate(X_list, axis=0).astype(np.float32)
    Y_seq = np.concatenate(Y_list, axis=0).astype(np.float32)

    # Train MLP-50ms
    log.info("\nTraining MLP-50ms...")
    H_tr, H_un, out_pred, test_r2, w_bins = _train_mlp_architecture(
        X_seq, Y_seq, 50, dt_ms=10.0, device='cuda')

    log.info("  Output R2: %.3f, window_bins: %d", test_r2, w_bins)
    log.info("  H_trained shape: %s", H_tr.shape)

    # Check dead neurons BEFORE filtering
    dead_trained = np.sum(np.std(H_tr, axis=0) <= 1e-10)
    dead_untrained = np.sum(np.std(H_un, axis=0) <= 1e-10)
    log.info("  Dead neurons BEFORE filter: trained=%d, untrained=%d (of %d)",
             dead_trained, dead_untrained, H_tr.shape[1])

    # Apply INTERSECTION filter
    var_trained = np.std(H_tr, axis=0)
    var_untrained = np.std(H_un, axis=0)
    alive_both = (var_trained > 1e-10) & (var_untrained > 1e-10)
    n_kept = int(alive_both.sum())
    log.info("  INTERSECTION filter: keeping %d/%d columns", n_kept, H_tr.shape[1])

    H_tr_filt = H_tr[:, alive_both]
    H_un_filt = H_un[:, alive_both]

    # Align with original bio targets
    offset = w_bins - 1
    n_mlp = H_tr_filt.shape[0]
    n_orig = session_data['bio_targets'].shape[0]
    n = min(n_mlp, n_orig - offset)

    log.info("  Aligned: %d samples (offset=%d)", n, offset)

    # Build session for Phase 3
    arch_session = {
        'H_trained': H_tr_filt[:n].astype(np.float64),
        'H_untrained': H_un_filt[:n].astype(np.float64),
        'bio_targets': session_data['bio_targets'][offset:offset + n],
        'bio_names': session_data['bio_names'],
        'epoch_mask': session_data['epoch_mask'][offset:offset + n],
        'trial_groups': session_data['trial_groups'][offset:offset + n],
        'meta': session_data['meta'],
        'model_info': session_data['model_info'],
        'hidden_dim': n_kept,
    }

    # Run Phase 3 probing on JUST firing_rate_input for speed
    log.info("\n--- Probing firing_rate_input only (quick test) ---")
    bio_names = arch_session['bio_names']
    fri_idx = bio_names.index('firing_rate_input') if 'firing_rate_input' in bio_names else 0
    target = arch_session['bio_targets'][:, fri_idx]
    groups = arch_session['trial_groups']

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score

    gkf = GroupKFold(n_splits=5)

    r2_t = float(np.mean(cross_val_score(
        Ridge(1.0), arch_session['H_trained'], target,
        cv=gkf, groups=groups, scoring='r2')))
    r2_u = float(np.mean(cross_val_score(
        Ridge(1.0), arch_session['H_untrained'], target,
        cv=gkf, groups=groups, scoring='r2')))
    dr2 = r2_t - r2_u

    log.info("  firing_rate_input:")
    log.info("    R2_trained:   %.4f", r2_t)
    log.info("    R2_untrained: %.4f", r2_u)
    log.info("    delta_R2:     %.4f", dr2)

    if dr2 < -0.1:
        log.warning("  STILL NEGATIVE (%.4f) — another problem beyond sparsity", dr2)
    elif dr2 > 0.02:
        log.info("  POSITIVE — intersection filter fixed the sparsity asymmetry!")
    else:
        log.info("  NEAR ZERO — MLP genuinely doesn't encode firing_rate_input")

    # Also check theta_gamma_pac
    if 'theta_gamma_pac' in bio_names:
        pac_idx = bio_names.index('theta_gamma_pac')
        pac_target = arch_session['bio_targets'][:, pac_idx]
        r2_t_pac = float(np.mean(cross_val_score(
            Ridge(1.0), arch_session['H_trained'], pac_target,
            cv=gkf, groups=groups, scoring='r2')))
        r2_u_pac = float(np.mean(cross_val_score(
            Ridge(1.0), arch_session['H_untrained'], pac_target,
            cv=gkf, groups=groups, scoring='r2')))
        dr2_pac = r2_t_pac - r2_u_pac
        log.info("  theta_gamma_pac:")
        log.info("    R2_trained:   %.4f", r2_t_pac)
        log.info("    R2_untrained: %.4f", r2_u_pac)
        log.info("    delta_R2:     %.4f", dr2_pac)

    log.info("\n" + "=" * 60)
    log.info("QUICK TEST COMPLETE")
    log.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
