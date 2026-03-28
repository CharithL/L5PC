#!/usr/bin/env python3
"""
run_kyzar_phase3_4.py

DESCARTES Circuit 6 Phases 3-4: Core probing + resample ablation.

Phase 3: Ridge delta-R2 with GroupKFold by trial (fixes temporal leakage),
         MLP probing, gate-specific probing on all 18 bio variables.
Phase 4: Resample ablation on mandatory candidates (delta-R2 > 0.05),
         epoch-specific ablation (fixation/encoding/maintenance/probe/response).

Requires Phase 2 output: trained models + extracted hidden states.
"""

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.neural_network import MLPRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('phase3_4')

EPOCH_NAMES = {0: 'fixation', 1: 'encoding', 2: 'maintenance', 3: 'probe', 4: 'response'}


def _convert_numpy(obj):
    if isinstance(obj, dict):
        return {str(k): _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =========================================================================
# DATA LOADING
# =========================================================================

def load_session(processed_dir, model_dir, subject, hidden_dim=64):
    """Load processed data + Phase 2 model outputs for one session."""
    sess_name = f'session_sub{subject}_ses2'
    data_dir = Path(processed_dir) / sess_name
    mdl_dir = Path(model_dir) / sess_name / f'lstm_h{hidden_dim}'

    if not mdl_dir.exists():
        return None

    # Bio targets and variable names
    with open(data_dir / 'bio_variable_names.json') as f:
        bio_names = json.load(f)
    bio_data = dict(np.load(data_dir / 'bio_targets.npz'))
    epoch_data = dict(np.load(data_dir / 'epoch_masks.npz'))
    meta = json.load(open(data_dir / 'metadata.json'))

    # Phase 2 outputs: hidden states
    hidden_trained = dict(np.load(mdl_dir / 'hidden_trained.npz'))
    hidden_untrained = dict(np.load(mdl_dir / 'hidden_untrained.npz'))

    # Load model for resample ablation
    model_path = mdl_dir / 'trained_model.pt'
    model_info = json.load(open(mdl_dir / 'output_validation.json'))

    n_trials = meta['n_trials']

    # Flatten all trials into single matrices, tracking trial boundaries
    h_trained_list, h_untrained_list, bio_list, epoch_list = [], [], [], []
    trial_ids = []

    for ti in range(n_trials):
        key = f'trial_{ti}'
        if key not in hidden_trained or key not in bio_data:
            continue

        ht = hidden_trained[key]   # (T_i, hidden_dim)
        hu = hidden_untrained[key] # (T_i, hidden_dim)
        bt = bio_data[key]         # (T_i, 18)
        ep = epoch_data[key]       # (T_i,)

        # Ensure matching lengths
        min_len = min(len(ht), len(hu), len(bt), len(ep))
        h_trained_list.append(ht[:min_len])
        h_untrained_list.append(hu[:min_len])
        bio_list.append(bt[:min_len])
        epoch_list.append(ep[:min_len])
        trial_ids.extend([ti] * min_len)

    if not h_trained_list:
        return None

    H_trained = np.concatenate(h_trained_list, axis=0).astype(np.float64)
    H_untrained = np.concatenate(h_untrained_list, axis=0).astype(np.float64)
    bio_targets = np.concatenate(bio_list, axis=0).astype(np.float64)
    epoch_mask = np.concatenate(epoch_list, axis=0).astype(int)
    trial_groups = np.array(trial_ids)

    return {
        'H_trained': H_trained,
        'H_untrained': H_untrained,
        'bio_targets': bio_targets,
        'bio_names': bio_names,
        'epoch_mask': epoch_mask,
        'trial_groups': trial_groups,
        'meta': meta,
        'model_info': model_info,
        'model_path': model_path,
        'data_dir': data_dir,
        'model_dir': mdl_dir,
        'hidden_dim': hidden_dim,
        'n_trials': n_trials,
    }


# =========================================================================
# PHASE 3: CORE PROBING
# =========================================================================

def ridge_delta_r2(H_trained, H_untrained, target, groups, alpha=1.0):
    """Ridge probe with GroupKFold by trial. Returns delta-R2."""
    gkf = GroupKFold(n_splits=5)

    # Check target has variance
    if np.std(target) < 1e-10:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0}

    r2_t = float(np.mean(cross_val_score(
        Ridge(alpha), H_trained, target, cv=gkf, groups=groups, scoring='r2',
        n_jobs=-1)))
    r2_u = float(np.mean(cross_val_score(
        Ridge(alpha), H_untrained, target, cv=gkf, groups=groups, scoring='r2',
        n_jobs=-1)))

    return {'r2_trained': r2_t, 'r2_untrained': r2_u, 'delta_r2': r2_t - r2_u}


def mlp_delta_r2(H_trained, H_untrained, target, groups):
    """MLP probe with GroupKFold by trial. Captures nonlinear encoding."""
    gkf = GroupKFold(n_splits=5)

    if np.std(target) < 1e-10:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0}

    # Sanitize inputs — NaN/Inf from dead neurons cause MLP explosions
    H_tr = np.nan_to_num(H_trained, nan=0.0, posinf=0.0, neginf=0.0)
    H_un = np.nan_to_num(H_untrained, nan=0.0, posinf=0.0, neginf=0.0)
    tgt = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                       early_stopping=True, random_state=42)
    r2_t = float(np.mean(cross_val_score(
        mlp, H_tr, tgt, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

    mlp_u = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         early_stopping=True, random_state=42)
    r2_u = float(np.mean(cross_val_score(
        mlp_u, H_un, tgt, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

    # Clamp to sane range — R² outside [-2, 1] indicates numerical garbage
    r2_t = float(np.clip(r2_t, -2.0, 1.0))
    r2_u = float(np.clip(r2_u, -2.0, 1.0))

    return {'r2_trained': r2_t, 'r2_untrained': r2_u, 'delta_r2': r2_t - r2_u}


def epoch_specific_probe(H_trained, H_untrained, target, epoch_mask,
                         trial_groups):
    """Probe within each Sternberg epoch separately."""
    epoch_results = {}
    for epoch_code, epoch_name in EPOCH_NAMES.items():
        mask = epoch_mask == epoch_code
        n_samples = np.sum(mask)
        if n_samples < 200:  # need minimum samples
            epoch_results[epoch_name] = {
                'n_samples': int(n_samples), 'delta_r2': 0.0, 'skipped': True}
            continue

        ht_ep = H_trained[mask]
        hu_ep = H_untrained[mask]
        tgt_ep = target[mask]
        grp_ep = trial_groups[mask]

        # Need enough unique groups for 5-fold
        n_unique = len(np.unique(grp_ep))
        n_splits = min(5, n_unique)
        if n_splits < 2:
            epoch_results[epoch_name] = {
                'n_samples': int(n_samples), 'delta_r2': 0.0, 'skipped': True}
            continue

        gkf = GroupKFold(n_splits=n_splits)
        if np.std(tgt_ep) < 1e-10:
            epoch_results[epoch_name] = {
                'n_samples': int(n_samples), 'delta_r2': 0.0, 'constant': True}
            continue

        r2_t = float(np.mean(cross_val_score(
            Ridge(1.0), ht_ep, tgt_ep, cv=gkf, groups=grp_ep, scoring='r2', n_jobs=-1)))
        r2_u = float(np.mean(cross_val_score(
            Ridge(1.0), hu_ep, tgt_ep, cv=gkf, groups=grp_ep, scoring='r2', n_jobs=-1)))

        epoch_results[epoch_name] = {
            'n_samples': int(n_samples),
            'r2_trained': r2_t,
            'r2_untrained': r2_u,
            'delta_r2': r2_t - r2_u,
        }
    return epoch_results


# =========================================================================
# iAAFT SCREENING (for MLP architectures — replaces ΔR² with R²_trained)
# =========================================================================

def iaaft_surrogate(signal, n_iterations=100, rng=None):
    """Generate an iAAFT surrogate preserving power spectrum + amplitude dist."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(signal)
    fft_orig = np.fft.rfft(signal)
    amplitudes = np.abs(fft_orig)
    sorted_original = np.sort(signal)

    # Initial: random phase
    random_phases = rng.uniform(0, 2 * np.pi, size=len(fft_orig))
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0

    surrogate = np.fft.irfft(amplitudes * np.exp(1j * random_phases), n=n)

    for _ in range(n_iterations):
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_original[rank_order]
        fft_surr = np.fft.rfft(surrogate)
        phases_surr = np.angle(fft_surr)
        surrogate = np.fft.irfft(amplitudes * np.exp(1j * phases_surr), n=n)

    return surrogate


def iaaft_screen_r2_trained(H_trained, target, groups, n_surrogates=50,
                             alpha=1.0, random_state=42, r2_floor=0.01):
    """Screen R²_trained against iAAFT null distribution with dual gate.

    For MLP architectures, ΔR² (trained - untrained) is inappropriate because
    untrained MLPs are random linear projections that preserve input statistics.
    Instead, we apply a DUAL screening gate:
      1. Statistical: R²_trained > 95th percentile of iAAFT null (p < 0.05)
      2. Effect size: R²_trained > r2_floor (default 0.01)

    Both conditions must be met to pass screening. The iAAFT test controls for
    temporal autocorrelation; the R² floor prevents trivially small effect sizes
    from reaching the expensive resample ablation step.

    Returns
    -------
    dict with keys:
        r2_trained : float — observed R² on real target
        iaaft_threshold : float — 95th percentile of iAAFT null
        iaaft_p : float — p-value (fraction of surrogates >= observed)
        r2_floor : float — minimum R² threshold for effect size gate
        passes_iaaft : bool — True if r2_trained > 95th percentile (p < 0.05)
        passes_r2_floor : bool — True if r2_trained > r2_floor
        passes_screen : bool — True if BOTH gates passed
        null_r2s : list — surrogate R² values for transparency
    """
    rng = np.random.default_rng(random_state)

    _fail = {
        'r2_trained': 0.0, 'iaaft_threshold': 0.0, 'iaaft_p': 1.0,
        'r2_floor': r2_floor, 'passes_iaaft': False,
        'passes_r2_floor': False, 'passes_screen': False, 'null_r2s': [],
    }

    if np.std(target) < 1e-10:
        return _fail

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return _fail

    gkf = GroupKFold(n_splits=n_splits)

    # Observed R²_trained on real target
    r2_trained = float(np.mean(cross_val_score(
        Ridge(alpha), H_trained, target, cv=gkf, groups=groups,
        scoring='r2', n_jobs=-1)))

    # Build iAAFT null: for each surrogate target, compute R²
    null_r2s = []
    for si in range(n_surrogates):
        surr_target = iaaft_surrogate(target, rng=rng)
        if np.std(surr_target) < 1e-10:
            null_r2s.append(0.0)
            continue
        surr_r2 = float(np.mean(cross_val_score(
            Ridge(alpha), H_trained, surr_target, cv=gkf, groups=groups,
            scoring='r2', n_jobs=-1)))
        null_r2s.append(surr_r2)

    null_r2s = np.array(null_r2s)
    threshold_95 = float(np.percentile(null_r2s, 95))
    p_value = float(np.mean(null_r2s >= r2_trained))

    passes_iaaft = r2_trained > threshold_95
    passes_r2_floor = r2_trained > r2_floor

    return {
        'r2_trained': r2_trained,
        'iaaft_threshold': threshold_95,
        'iaaft_p': p_value,
        'r2_floor': r2_floor,
        'passes_iaaft': passes_iaaft,
        'passes_r2_floor': passes_r2_floor,
        'passes_screen': passes_iaaft and passes_r2_floor,  # DUAL GATE
        'null_r2s': null_r2s.tolist(),
    }


def run_phase3_iaaft(session_data, n_surrogates=50):
    """Phase 3 for MLP architectures: iAAFT screening on R²_trained.

    Replaces ΔR² screening with iAAFT-based screening. Variables where
    R²_trained > 95th percentile of iAAFT null proceed to Phase 4.
    """
    H_t = session_data['H_trained']
    bio = session_data['bio_targets']
    names = session_data['bio_names']
    groups = session_data['trial_groups']
    epoch_mask = session_data['epoch_mask']

    results = {}

    for vi, vname in enumerate(names):
        target = bio[:, vi]
        log.info("    [iAAFT] Probing %s (%d/%d)...", vname, vi + 1, len(names))

        # iAAFT screening on R²_trained (replaces ΔR² for MLPs)
        iaaft = iaaft_screen_r2_trained(H_t, target, groups,
                                         n_surrogates=n_surrogates)

        # Also compute raw R²_trained for epoch-specific (no untrained needed)
        # Epoch-specific probing uses R²_trained only
        epoch_results = {}
        for epoch_code, epoch_name in EPOCH_NAMES.items():
            mask = epoch_mask == epoch_code
            n_samples = np.sum(mask)
            if n_samples < 200:
                epoch_results[epoch_name] = {
                    'n_samples': int(n_samples), 'r2_trained': 0.0, 'skipped': True}
                continue
            ht_ep = H_t[mask]
            tgt_ep = target[mask]
            grp_ep = groups[mask]
            n_unique = len(np.unique(grp_ep))
            n_splits = min(5, n_unique)
            if n_splits < 2 or np.std(tgt_ep) < 1e-10:
                epoch_results[epoch_name] = {
                    'n_samples': int(n_samples), 'r2_trained': 0.0, 'skipped': True}
                continue
            gkf = GroupKFold(n_splits=n_splits)
            r2_t = float(np.mean(cross_val_score(
                Ridge(1.0), ht_ep, tgt_ep, cv=gkf, groups=grp_ep,
                scoring='r2', n_jobs=-1)))
            epoch_results[epoch_name] = {
                'n_samples': int(n_samples), 'r2_trained': r2_t}

        # Raw correlation profile
        hdim = H_t.shape[1]
        n_check = min(hdim, 128)
        corrs = [abs(np.corrcoef(H_t[:, d], target)[0, 1])
                 for d in range(n_check)]
        max_r = float(max(corrs)) if corrs else 0.0
        n_corr_dims = int(sum(1 for c in corrs if c > 0.3))

        results[vname] = {
            'iaaft': iaaft,
            'epoch_specific': epoch_results,
            'max_abs_r': max_r,
            'n_dims_above_0.3': n_corr_dims,
            # Compatibility keys for logging
            'ridge': {
                'r2_trained': iaaft['r2_trained'],
                'r2_untrained': 0.0,  # not computed for MLPs
                'delta_r2': iaaft['r2_trained'],  # for display only
            },
            'mlp': {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0},
        }

        if iaaft['passes_screen']:
            status = "PASS (both gates)"
        elif iaaft.get('passes_iaaft') and not iaaft.get('passes_r2_floor'):
            status = "FAIL (R²<%.3f)" % iaaft.get('r2_floor', 0.01)
        elif iaaft.get('passes_r2_floor') and not iaaft.get('passes_iaaft'):
            status = "FAIL (p=%.3f)" % iaaft['iaaft_p']
        else:
            status = "FAIL (both gates)"
        log.info("      R2_trained=%.4f  iAAFT_95th=%.4f  p=%.3f  [%s]  max|r|=%.3f",
                 iaaft['r2_trained'], iaaft['iaaft_threshold'],
                 iaaft['iaaft_p'], status, max_r)

    return results


def run_phase4_iaaft(session_data, phase3_iaaft_results):
    """Phase 4 for MLP architectures: resample ablation on iAAFT-screened variables.

    Variables proceed to ablation if R²_trained > iAAFT 95th percentile
    (instead of the ΔR² > 0.05 threshold used for LSTMs).
    """
    H_t = session_data['H_trained']
    bio = session_data['bio_targets']
    names = session_data['bio_names']
    groups = session_data['trial_groups']
    epoch_mask = session_data['epoch_mask']

    # Select candidates: variables that passed iAAFT screening
    candidates = []
    for vi, vname in enumerate(names):
        p3 = phase3_iaaft_results.get(vname, {})
        iaaft = p3.get('iaaft', {})
        if iaaft.get('passes_screen', False):
            candidates.append((vi, vname))

    log.info("  Phase 4 (iAAFT): %d/%d variables passed iAAFT screening",
             len(candidates), len(names))

    results = {}
    rng = np.random.default_rng(42)

    for vi, vname in candidates:
        target = bio[:, vi]
        log.info("    Ablating %s...", vname)

        # Full resample ablation (same as LSTM path)
        abl = resample_ablation(H_t, target, groups, rng=rng)

        # Epoch-specific ablation
        ep_abl = epoch_specific_ablation(
            H_t, target, epoch_mask, groups, rng=rng)

        results[vname] = {
            'resample_ablation': abl,
            'epoch_ablation': ep_abl,
        }

        log.info("      Overall: %s (causal at %d k-values)",
                 abl['overall_verdict'], abl['n_causal_k'])
        for ep_name, ep_res in ep_abl.items():
            if not ep_res.get('skipped'):
                log.info("        %s: z=%.2f %s",
                         ep_name, ep_res.get('z_score', 0),
                         ep_res.get('verdict', ''))

    return results


def run_phase3(session_data):
    """Run all Phase 3 probes on one session."""
    H_t = session_data['H_trained']
    H_u = session_data['H_untrained']
    bio = session_data['bio_targets']
    names = session_data['bio_names']
    groups = session_data['trial_groups']
    epoch_mask = session_data['epoch_mask']

    results = {}

    for vi, vname in enumerate(names):
        target = bio[:, vi]
        log.info("    Probing %s (%d/%d)...", vname, vi + 1, len(names))

        # Ridge delta-R2 (main probe)
        ridge = ridge_delta_r2(H_t, H_u, target, groups)

        # MLP delta-R2 (nonlinear check)
        mlp = mlp_delta_r2(H_t, H_u, target, groups)

        # Epoch-specific probing
        epoch = epoch_specific_probe(H_t, H_u, target, epoch_mask, groups)

        # Raw correlation profile
        hdim = H_t.shape[1]
        n_check = min(hdim, 128)
        corrs = [abs(np.corrcoef(H_t[:, d], target)[0, 1])
                 for d in range(n_check)]
        max_r = float(max(corrs)) if corrs else 0.0
        n_corr_dims = int(sum(1 for c in corrs if c > 0.3))

        results[vname] = {
            'ridge': ridge,
            'mlp': mlp,
            'epoch_specific': epoch,
            'max_abs_r': max_r,
            'n_dims_above_0.3': n_corr_dims,
        }

        log.info("      Ridge dR2=%.3f  MLP dR2=%.3f  max|r|=%.3f  dims>0.3=%d",
                 ridge['delta_r2'], mlp['delta_r2'], max_r, n_corr_dims)

        # Log epoch results for interesting variables
        for ep_name, ep_res in epoch.items():
            if not ep_res.get('skipped') and not ep_res.get('constant'):
                dr2 = ep_res.get('delta_r2', 0)
                if abs(dr2) > 0.02:
                    log.info("        %s: dR2=%.3f", ep_name, dr2)

    return results


# =========================================================================
# PHASE 4: RESAMPLE ABLATION
# =========================================================================

def resample_ablation(H_trained, target, groups, k_values=None,
                      n_resamples=20, rng=None):
    """
    Resample ablation: zero out k highest-correlated dims,
    measure degradation vs random k dims.
    Uses GroupKFold for all Ridge evaluations.
    """
    if k_values is None:
        hdim = H_trained.shape[1]
        k_values = [max(1, int(hdim * f)) for f in [0.05, 0.10, 0.20, 0.30, 0.40]]
        k_values = sorted(set(k_values))

    if rng is None:
        rng = np.random.default_rng(42)

    gkf = GroupKFold(n_splits=5)

    # Baseline R2
    baseline_r2 = float(np.mean(cross_val_score(
        Ridge(1.0), H_trained, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

    # Per-dim correlation with target
    hdim = H_trained.shape[1]
    dim_corrs = np.array([abs(np.corrcoef(H_trained[:, d], target)[0, 1])
                          for d in range(hdim)])

    results_per_k = {}
    for k in k_values:
        # Targeted ablation: zero out top-k correlated dims
        top_k = np.argsort(dim_corrs)[-k:]
        H_ablated = H_trained.copy()
        H_ablated[:, top_k] = 0.0

        ablated_r2 = float(np.mean(cross_val_score(
            Ridge(1.0), H_ablated, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))
        degradation = baseline_r2 - ablated_r2

        # Random ablation control
        rand_r2s = []
        for _ in range(n_resamples):
            rand_dims = rng.choice(hdim, k, replace=False)
            H_rand = H_trained.copy()
            H_rand[:, rand_dims] = 0.0
            rr2 = float(np.mean(cross_val_score(
                Ridge(1.0), H_rand, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))
            rand_r2s.append(rr2)

        rand_mean = float(np.mean(rand_r2s))
        rand_std = float(np.std(rand_r2s)) + 1e-10
        z_score = (ablated_r2 - rand_mean) / rand_std
        verdict = 'CAUSAL' if z_score < -2.0 else 'BYPRODUCT'

        results_per_k[k] = {
            'k': k,
            'k_frac': float(k / hdim),
            'baseline_r2': baseline_r2,
            'ablated_r2': ablated_r2,
            'degradation': degradation,
            'rand_mean': rand_mean,
            'rand_std': rand_std,
            'z_score': z_score,
            'verdict': verdict,
        }

    # Overall: causal if z < -2 at any k
    n_causal = sum(1 for r in results_per_k.values() if r['verdict'] == 'CAUSAL')

    return {
        'baseline_r2': baseline_r2,
        'per_k': results_per_k,
        'n_causal_k': n_causal,
        'overall_verdict': 'MANDATORY' if n_causal >= 2 else
                           'CAUSAL' if n_causal >= 1 else 'BYPRODUCT',
    }


def epoch_specific_ablation(H_trained, target, epoch_mask, trial_groups,
                            k_frac=0.20, n_resamples=10, rng=None):
    """Resample ablation within each epoch. Finds phase-locked mandatory variables."""
    if rng is None:
        rng = np.random.default_rng(42)

    hdim = H_trained.shape[1]
    k = max(1, int(hdim * k_frac))

    results = {}
    for epoch_code, epoch_name in EPOCH_NAMES.items():
        mask = epoch_mask == epoch_code
        n_samples = np.sum(mask)
        if n_samples < 200:
            results[epoch_name] = {'skipped': True, 'n_samples': int(n_samples)}
            continue

        ht_ep = H_trained[mask]
        tgt_ep = target[mask]
        grp_ep = trial_groups[mask]

        n_unique = len(np.unique(grp_ep))
        n_splits = min(5, n_unique)
        if n_splits < 2 or np.std(tgt_ep) < 1e-10:
            results[epoch_name] = {'skipped': True, 'n_samples': int(n_samples)}
            continue

        gkf = GroupKFold(n_splits=n_splits)

        # Baseline
        base_r2 = float(np.mean(cross_val_score(
            Ridge(1.0), ht_ep, tgt_ep, cv=gkf, groups=grp_ep, scoring='r2', n_jobs=-1)))

        # Targeted ablation
        dim_corrs = np.array([abs(np.corrcoef(ht_ep[:, d], tgt_ep)[0, 1])
                              for d in range(hdim)])
        top_k = np.argsort(dim_corrs)[-k:]
        H_abl = ht_ep.copy()
        H_abl[:, top_k] = 0.0
        abl_r2 = float(np.mean(cross_val_score(
            Ridge(1.0), H_abl, tgt_ep, cv=gkf, groups=grp_ep, scoring='r2', n_jobs=-1)))

        # Random control
        rand_r2s = []
        for _ in range(n_resamples):
            rd = rng.choice(hdim, k, replace=False)
            H_r = ht_ep.copy()
            H_r[:, rd] = 0.0
            rr2 = float(np.mean(cross_val_score(
                Ridge(1.0), H_r, tgt_ep, cv=gkf, groups=grp_ep, scoring='r2', n_jobs=-1)))
            rand_r2s.append(rr2)

        rand_mean = float(np.mean(rand_r2s))
        rand_std = float(np.std(rand_r2s)) + 1e-10
        z = (abl_r2 - rand_mean) / rand_std

        results[epoch_name] = {
            'n_samples': int(n_samples),
            'baseline_r2': base_r2,
            'ablated_r2': abl_r2,
            'degradation': base_r2 - abl_r2,
            'z_score': z,
            'verdict': 'CAUSAL' if z < -2.0 else 'BYPRODUCT',
        }

    return results


def run_phase4(session_data, phase3_results):
    """Run resample ablation on mandatory candidates."""
    H_t = session_data['H_trained']
    bio = session_data['bio_targets']
    names = session_data['bio_names']
    groups = session_data['trial_groups']
    epoch_mask = session_data['epoch_mask']

    # Select candidates: ridge or MLP delta-R2 > 0.05
    candidates = []
    for vi, vname in enumerate(names):
        p3 = phase3_results.get(vname, {})
        ridge_dr2 = p3.get('ridge', {}).get('delta_r2', 0)
        mlp_dr2 = p3.get('mlp', {}).get('delta_r2', 0)
        if ridge_dr2 > 0.05 or mlp_dr2 > 0.05:
            candidates.append((vi, vname))

    log.info("  Phase 4: %d/%d variables above dR2 > 0.05 threshold",
             len(candidates), len(names))

    results = {}
    rng = np.random.default_rng(42)

    for vi, vname in candidates:
        target = bio[:, vi]
        log.info("    Ablating %s...", vname)

        # Full resample ablation
        abl = resample_ablation(H_t, target, groups, rng=rng)

        # Epoch-specific ablation
        ep_abl = epoch_specific_ablation(
            H_t, target, epoch_mask, groups, rng=rng)

        results[vname] = {
            'resample_ablation': abl,
            'epoch_ablation': ep_abl,
        }

        log.info("      Overall: %s (causal at %d k-values)",
                 abl['overall_verdict'], abl['n_causal_k'])
        for ep_name, ep_res in ep_abl.items():
            if not ep_res.get('skipped'):
                log.info("        %s: z=%.2f %s",
                         ep_name, ep_res.get('z_score', 0),
                         ep_res.get('verdict', ''))

    return results


# =========================================================================
# MAIN
# =========================================================================

def process_session(processed_dir, model_dir, subject, hidden_dim, output_dir):
    """Run Phase 3 + 4 on one session."""
    log.info("=" * 70)
    log.info("SESSION: sub-%s (h=%d)", subject, hidden_dim)
    log.info("=" * 70)

    session_data = load_session(processed_dir, model_dir, subject, hidden_dim)
    if session_data is None:
        log.warning("  No Phase 2 output for sub-%s h=%d, skipping", subject, hidden_dim)
        return None

    cc = session_data['model_info'].get('test_cc', 0)
    log.info("  Samples: %d, Trials: %d, Hidden: %d, CC: %.3f",
             len(session_data['H_trained']), session_data['n_trials'],
             hidden_dim, cc)

    # Phase 3
    log.info("  --- Phase 3: Core Probing ---")
    t0 = time.time()
    p3 = run_phase3(session_data)
    t3 = time.time() - t0
    log.info("  Phase 3 done (%.1fs)", t3)

    # Phase 4
    log.info("  --- Phase 4: Resample Ablation ---")
    t0 = time.time()
    p4 = run_phase4(session_data, p3)
    t4 = time.time() - t0
    log.info("  Phase 4 done (%.1fs)", t4)

    # Summary
    n_mandatory = sum(1 for v in p4.values()
                      if v['resample_ablation']['overall_verdict'] == 'MANDATORY')
    n_causal = sum(1 for v in p4.values()
                   if v['resample_ablation']['overall_verdict'] in ('MANDATORY', 'CAUSAL'))
    n_tested = len(p4)

    # Epoch-specific mandatory check
    epoch_mandatory = {}
    for vname, vres in p4.items():
        for ep_name, ep_res in vres.get('epoch_ablation', {}).items():
            if ep_res.get('verdict') == 'CAUSAL':
                epoch_mandatory.setdefault(ep_name, []).append(vname)

    log.info("\n  SUMMARY: sub-%s h=%d CC=%.3f", subject, hidden_dim, cc)
    log.info("    Mandatory: %d/%d tested (%d candidates from Phase 3)",
             n_mandatory, n_tested, n_tested)
    log.info("    Causal (any k): %d/%d", n_causal, n_tested)
    if epoch_mandatory:
        for ep, vars_list in epoch_mandatory.items():
            log.info("    Epoch %s mandatory: %s", ep, ', '.join(vars_list))

    result = {
        'subject': subject,
        'hidden_dim': hidden_dim,
        'output_cc': cc,
        'n_samples': len(session_data['H_trained']),
        'n_trials': session_data['n_trials'],
        'phase3': p3,
        'phase4': p4,
        'summary': {
            'n_mandatory': n_mandatory,
            'n_causal': n_causal,
            'n_tested': n_tested,
            'epoch_mandatory': epoch_mandatory,
        },
        'time_phase3': t3,
        'time_phase4': t4,
    }

    # Save per-session
    sess_out = Path(output_dir) / f'sub{subject}_h{hidden_dim}'
    sess_out.mkdir(parents=True, exist_ok=True)
    with open(sess_out / 'phase3_4_results.json', 'w') as f:
        json.dump(_convert_numpy(result), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Kyzar Phase 3-4: Probing + Ablation')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', default='results/kyzar_phase3_4')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--subjects', nargs='*', default=None,
                        help='Specific subjects (default: all available)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sessions with Phase 2 output
    model_root = Path(args.model_dir)
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = []
        for d in sorted(model_root.iterdir()):
            if d.is_dir() and d.name.startswith('session_sub'):
                h_dir = d / f'lstm_h{args.hidden_dim}'
                if h_dir.exists() and (h_dir / 'hidden_trained.npz').exists():
                    sub = d.name.replace('session_sub', '').replace('_ses2', '')
                    subjects.append(sub)

    log.info("Phase 3-4: %d sessions, h=%d", len(subjects), args.hidden_dim)
    t_start = time.time()

    all_results = []
    for sub in subjects:
        result = process_session(
            args.processed_dir, args.model_dir, sub,
            args.hidden_dim, str(output_dir))
        if result is not None:
            all_results.append(result)

    # Cross-session summary
    log.info("\n" + "=" * 70)
    log.info("CROSS-SESSION SUMMARY")
    log.info("=" * 70)
    log.info("%-8s %5s %6s %6s %6s %8s",
             'Subject', 'CC', '#Mand', '#Caus', '#Test', 'EpochMand')
    log.info("-" * 50)

    for r in sorted(all_results, key=lambda x: x['output_cc'], reverse=True):
        s = r['summary']
        ep_str = '; '.join(f"{e}:{len(v)}" for e, v in s['epoch_mandatory'].items()) \
                 if s['epoch_mandatory'] else 'none'
        log.info("sub-%-4s %5.3f %6d %6d %6d %8s",
                 r['subject'], r['output_cc'],
                 s['n_mandatory'], s['n_causal'], s['n_tested'], ep_str)

    # Save cross-session
    with open(output_dir / 'cross_session_summary.json', 'w') as f:
        json.dump(_convert_numpy({
            'n_sessions': len(all_results),
            'hidden_dim': args.hidden_dim,
            'sessions': [{
                'subject': r['subject'],
                'output_cc': r['output_cc'],
                'n_mandatory': r['summary']['n_mandatory'],
                'n_causal': r['summary']['n_causal'],
                'n_tested': r['summary']['n_tested'],
                'epoch_mandatory': r['summary']['epoch_mandatory'],
            } for r in all_results],
            'total_time_min': (time.time() - t_start) / 60,
        }), f, indent=2)

    log.info("\nTotal time: %.1f min", (time.time() - t_start) / 60)
    log.info("Saved: %s", output_dir)


if __name__ == '__main__':
    main()
