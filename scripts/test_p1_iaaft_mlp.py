#!/usr/bin/env python3
"""Test: MLP-50ms on sub-2 with iAAFT screening (replaces ΔR² for MLPs)."""

import sys
import os
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('test_iaaft_mlp')

KYZAR_DIR = r"C:\Users\chari\OneDrive\Documents\Descartes_Cogito\Kyzar\data\kyzar_processed"
MODEL_DIR = r"C:\Users\chari\OneDrive\Documents\Descartes_Cogito\L5PC\models\kyzar"
SUBJECT = '2'


def main():
    log.info("=" * 60)
    log.info("TEST: MLP-50ms iAAFT screening on sub-%s", SUBJECT)
    log.info("  Method: R²_trained vs iAAFT null (not ΔR²)")
    log.info("=" * 60)

    # Import Phase 3-4 module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "kyzar_phase3_4",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_kyzar_phase3_4.py"))
    phase34 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase34)

    # Import MLP trainer
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

    # Apply INTERSECTION dead neuron filter
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

    # Build session for iAAFT Phase 3
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

    # Run iAAFT Phase 3 (50 surrogates for speed — use 200 for production)
    log.info("\n--- Phase 3: iAAFT screening (50 surrogates) ---")
    p3 = phase34.run_phase3_iaaft(arch_session, n_surrogates=50)

    # Run Phase 4 on iAAFT-screened variables
    log.info("\n--- Phase 4: Resample ablation on iAAFT-screened variables ---")
    p4 = phase34.run_phase4_iaaft(arch_session, p3)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("RESULTS: MLP-50ms iAAFT screening")
    log.info("=" * 60)

    log.info("\n%-25s | %8s | %8s | %6s | %s",
             "Variable", "R2_train", "iAAFT95", "p", "Screen")
    log.info("-" * 80)
    for vname, vres in p3.items():
        iaaft = vres.get('iaaft', {})
        status = "PASS" if iaaft.get('passes_screen') else "fail"
        log.info("%-25s | %8.4f | %8.4f | %6.3f | %s",
                 vname, iaaft.get('r2_trained', 0),
                 iaaft.get('iaaft_threshold', 0),
                 iaaft.get('iaaft_p', 1), status)

    # Phase 4 results
    if p4:
        log.info("\nPhase 4 ablation results:")
        for vname, vres in p4.items():
            abl = vres.get('resample_ablation', {})
            log.info("  %s: %s (causal at %d k-values)",
                     vname, abl.get('overall_verdict', '?'),
                     abl.get('n_causal_k', 0))
    else:
        log.info("\nNo variables passed iAAFT screening → no Phase 4 ablation")

    # Compare with old ΔR² method for reference
    log.info("\n--- For comparison: old ΔR² values (DO NOT USE for decision) ---")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score

    gkf = GroupKFold(n_splits=5)
    groups = arch_session['trial_groups']
    bio_names = arch_session['bio_names']
    for vi, vname in enumerate(bio_names[:5]):  # just first 5 for speed
        target = arch_session['bio_targets'][:, vi]
        if np.std(target) < 1e-10:
            continue
        r2_t = float(np.mean(cross_val_score(
            Ridge(1.0), arch_session['H_trained'], target,
            cv=gkf, groups=groups, scoring='r2')))
        r2_u = float(np.mean(cross_val_score(
            Ridge(1.0), arch_session['H_untrained'], target,
            cv=gkf, groups=groups, scoring='r2')))
        log.info("  %s: R2_tr=%.4f  R2_un=%.4f  ΔR2=%.4f  (old method, biased)",
                 vname, r2_t, r2_u, r2_t - r2_u)

    log.info("\n" + "=" * 60)
    log.info("TEST COMPLETE")
    log.info("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
