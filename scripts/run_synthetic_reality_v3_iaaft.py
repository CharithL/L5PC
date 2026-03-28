#!/usr/bin/env python3
"""
run_synthetic_reality_v3_iaaft.py

DESCARTES Synthetic Reality v3 — Unified iAAFT Screening

Replaces ΔR² screening from v2 with the unified iAAFT dual gate:
  Gate 1: R²_trained > 95th percentile of iAAFT null (p < 0.05)
  Gate 2: R²_trained > 0.01 (effect size floor)

Only runs the THREE conditions that passed v2 verification:
  1. SEQUENTIAL    — real Sternberg data (control)
  2. SPATIAL_ONLY  — preserve co-activation patterns, destroy timing
  3. PURE_NOISE    — Poisson noise matched to statistics (zombie control)

RATE_MODULATED and TEMPORAL_ONLY excluded: failed v2 verification
(rate_corr < 0.8 and rate_std_ratio > 0.2 respectively).

Predictions:
  SEQUENTIAL  → theta_gamma_pac mandatory (temporal structure drives PAC)
  SPATIAL_ONLY → gamma_power mandatory (spatial co-activation preserves instantaneous power)
  PURE_NOISE  → 0 mandatory (zombie)

If all conditions produce identical mandatory sets → variables are absolute
necessities of the architecture, not reality-dependent reflections.

Usage:
  python scripts/run_synthetic_reality_v3_iaaft.py \
    --processed-dir data/kyzar_processed \
    --output-dir results/synthetic_reality_v3 \
    --source-subject 5 \
    --device cpu
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy.signal import butter, filtfilt, hilbert
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('synthetic_reality_v3')

# Same 7 probe targets as v2
PROBE_VARIABLES = [
    'firing_rate_input', 'trial_variance', 'theta_power',
    'gamma_power', 'theta_gamma_pac',
    'population_synchrony', 'rate_covariance',
]

# Only the 3 verified conditions
VERIFIED_CONDITIONS = ['SEQUENTIAL', 'SPATIAL_ONLY', 'PURE_NOISE']


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
# MODEL (identical to v2)
# =========================================================================

class LSTMSurrogate(nn.Module):
    def __init__(self, n_in, n_out, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        dr = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_in, hidden_dim, n_layers,
                            batch_first=True, dropout=dr)
        self.output_layer = nn.Linear(hidden_dim, n_out)
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                         enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        pred = self.output_layer(out)
        return pred, out


# =========================================================================
# DATA LOADING (identical to v2)
# =========================================================================

def load_source_data(processed_dir, subject):
    sess_name = 'session_sub{}_ses2'.format(subject)
    data_dir = Path(processed_dir) / sess_name

    X_data = dict(np.load(data_dir / 'X_trials.npz'))
    Y_data = dict(np.load(data_dir / 'Y_trials.npz'))
    E_data = dict(np.load(data_dir / 'epoch_masks.npz'))

    with open(data_dir / 'metadata.json') as f:
        meta = json.load(f)

    n_trials = meta['n_trials']
    trials = []
    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key in X_data and key in Y_data:
            trials.append({
                'X': X_data[key].astype(np.float32),
                'Y': Y_data[key].astype(np.float32),
                'epoch': E_data[key] if key in E_data else np.zeros(X_data[key].shape[0], dtype=int),
                'meta': meta['trial_metadata'][ti] if ti < len(meta.get('trial_metadata', [])) else {},
            })

    log.info("Loaded %d trials from sub-%s", len(trials), subject)
    log.info("  n_in=%d, n_out=%d", trials[0]['X'].shape[1], trials[0]['Y'].shape[1])
    return trials, meta


# =========================================================================
# TRANSFORM VERIFICATION (identical to v2, but only for our 3 conditions)
# =========================================================================

def _bandpower(x, fs, fmin, fmax):
    T = len(x)
    if T < 20 or fs < 2 * fmax:
        return np.zeros(T)
    try:
        nyq = fs / 2
        low, high = max(fmin / nyq, 0.01), min(fmax / nyq, 0.99)
        if low >= high:
            return np.zeros(T)
        b, a = butter(3, [low, high], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        filtered = filtfilt(b, a, x, padlen=padlen)
        analytic = hilbert(filtered)
        return np.abs(analytic) ** 2
    except Exception:
        return np.zeros(T)


def verify_transform(original_trials, transformed_trials, condition_name, dt_ms=10):
    log.info("  VERIFICATION: checking %s transform...", condition_name)
    fs = 1000.0 / dt_ms

    orig_X = np.concatenate([t['X'] for t in original_trials], axis=0)
    trans_X = np.concatenate([t['X'] for t in transformed_trials], axis=0)

    n_neurons_orig = orig_X.shape[1]
    n_neurons_trans = trans_X.shape[1]

    verification = {
        'condition': condition_name,
        'n_neurons_orig': n_neurons_orig,
        'n_neurons_trans': n_neurons_trans,
    }

    # Mean firing rates per neuron
    orig_rates = np.mean(orig_X, axis=0)
    trans_rates = np.mean(trans_X, axis=0)[:n_neurons_orig] if n_neurons_trans >= n_neurons_orig else np.mean(trans_X, axis=0)
    min_n = min(len(orig_rates), len(trans_rates))
    rate_corr = float(np.corrcoef(orig_rates[:min_n], trans_rates[:min_n])[0, 1]) if min_n > 1 else 0.0
    verification['rate_correlation'] = rate_corr
    log.info("    Rate correlation: %.3f", rate_corr)

    # Spectral content
    orig_pop = np.mean(orig_X, axis=1)
    trans_pop = np.mean(trans_X, axis=1)
    theta_ratio = float(np.mean(_bandpower(trans_pop, fs, 4, 8))) / (float(np.mean(_bandpower(orig_pop, fs, 4, 8))) + 1e-10)
    verification['theta_ratio'] = theta_ratio
    log.info("    Theta power ratio: %.3f", theta_ratio)

    # Pairwise correlations
    n_sample = min(5000, len(orig_X), len(trans_X))
    n_check = min(min_n, 50)
    if n_check > 1:
        orig_corr_mat = np.corrcoef(orig_X[:n_sample, :n_check].T)
        trans_corr_mat = np.corrcoef(trans_X[:n_sample, :n_check].T)
        triu_idx = np.triu_indices(n_check, k=1)
        pairwise_corr = float(np.corrcoef(orig_corr_mat[triu_idx], trans_corr_mat[triu_idx])[0, 1])
    else:
        pairwise_corr = 0.0
    verification['pairwise_correlation'] = pairwise_corr
    log.info("    Pairwise correlation preservation: %.3f", pairwise_corr)

    # Condition-specific validation
    if condition_name == 'SPATIAL_ONLY':
        verification['preserves_coactivation'] = pairwise_corr > 0.5
        verification['destroys_timing'] = theta_ratio < 0.5
        verification['valid'] = verification['preserves_coactivation']
    elif condition_name == 'PURE_NOISE':
        verification['valid'] = True
    else:  # SEQUENTIAL
        verification['valid'] = True

    log.info("    Verification: %s", "PASS" if verification['valid'] else "FAIL")
    return verification


# =========================================================================
# CONDITION GENERATORS (from v2, only the 3 we need)
# =========================================================================

def condition_sequential(trials):
    log.info("  Condition: SEQUENTIAL (real Sternberg data, %d trials)", len(trials))
    return trials


def condition_spatial_only(trials, rng):
    """Preserve co-activation patterns, destroy timing via 200ms window shuffles."""
    log.info("  Condition: SPATIAL_ONLY (preserve co-activation, destroy timing)")
    dt_ms = 10
    window_size = int(200 / dt_ms)  # 20 timesteps

    new_trials = []
    for t in trials:
        X = t['X'].copy()
        Y = t['Y'].copy()
        T = X.shape[0]

        X_perm = X.copy()
        Y_perm = Y.copy()
        for start in range(0, T, window_size):
            end = min(start + window_size, T)
            perm = rng.permutation(end - start) + start
            X_perm[start:end] = X[perm]
            Y_perm[start:end] = Y[perm]

        new_trials.append({
            'X': X_perm.astype(np.float32),
            'Y': Y_perm.astype(np.float32),
            'epoch': t['epoch'].copy(),
            'meta': t.get('meta', {}),
        })

    log.info("    %d trials: timesteps permuted within %dms windows",
             len(new_trials), window_size * dt_ms)
    return new_trials


def condition_pure_noise(trials, rng):
    """Gaussian noise matched to real statistics."""
    log.info("  Condition: PURE_NOISE (Gaussian noise matched to statistics)")

    all_X = np.concatenate([t['X'] for t in trials], axis=0)
    all_Y = np.concatenate([t['Y'] for t in trials], axis=0)

    x_mean = np.mean(all_X, axis=0)
    x_std = np.std(all_X, axis=0)
    y_mean = np.mean(all_Y, axis=0)
    y_std = np.std(all_Y, axis=0)

    n_in = all_X.shape[1]
    n_out = all_Y.shape[1]

    noise_trials = []
    for t in trials:
        T = t['X'].shape[0]
        X_noise = rng.normal(0, 1, (T, n_in)) * x_std[None, :] + x_mean[None, :]
        X_noise = np.maximum(X_noise, 0).astype(np.float32)
        Y_noise = rng.normal(0, 1, (T, n_out)) * y_std[None, :] + y_mean[None, :]
        Y_noise = np.maximum(Y_noise, 0).astype(np.float32)
        epoch_noise = rng.integers(0, 5, size=T)
        noise_trials.append({
            'X': X_noise, 'Y': Y_noise,
            'epoch': epoch_noise, 'meta': {},
        })

    log.info("    Generated %d noise trials matched to real stats", len(noise_trials))
    return noise_trials


# =========================================================================
# BIO VARIABLE COMPUTATION (identical to v2)
# =========================================================================

@njit(cache=True)
def _rolling_var_numba(x, win):
    T = len(x)
    result = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        if hi - lo > 1:
            s = 0.0
            s2 = 0.0
            n = hi - lo
            for j in range(lo, hi):
                s += x[j]
                s2 += x[j] * x[j]
            mean = s / n
            result[i] = s2 / n - mean * mean
    return result


def _rolling_var(x, win):
    if HAS_NUMBA:
        return _rolling_var_numba(x, win)
    T = len(x)
    result = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        result[i] = np.var(x[lo:hi]) if hi - lo > 1 else 0
    return result


def _instantaneous_phase(x, fs, fmin, fmax):
    T = len(x)
    if T < 20 or fs < 2 * fmax:
        return np.zeros(T)
    try:
        nyq = fs / 2
        low, high = max(fmin / nyq, 0.01), min(fmax / nyq, 0.99)
        if low >= high:
            return np.zeros(T)
        b, a = butter(3, [low, high], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        filtered = filtfilt(b, a, x, padlen=padlen)
        return np.angle(hilbert(filtered))
    except Exception:
        return np.zeros(T)


def _pac_strength(x, fs, theta_range=(4, 8), gamma_range=(30, 80)):
    T = len(x)
    gmax = min(gamma_range[1], fs / 2 - 1)
    if T < 50 or fs < 2 * gmax:
        return np.zeros(T)
    try:
        theta_phase = _instantaneous_phase(x, fs, *theta_range)
        gamma_amp = np.sqrt(np.maximum(_bandpower(x, fs, gamma_range[0], gmax), 0))
        win = max(1, int(500 / (1000 / fs)))
        half = win // 2
        pac = np.zeros(T)
        for i in range(T):
            lo, hi = max(0, i - half), min(T, i + half + 1)
            if hi - lo < 5:
                continue
            z = np.mean(gamma_amp[lo:hi] * np.exp(1j * theta_phase[lo:hi]))
            pac[i] = np.abs(z)
        return pac
    except Exception:
        return np.zeros(T)


def _population_synchrony(X, win):
    T, n_neurons = X.shape
    if n_neurons < 2:
        return np.zeros(T)
    half = win // 2
    result = np.zeros(T)
    n_check = min(n_neurons, 30)
    cols = np.linspace(0, n_neurons - 1, n_check, dtype=int)
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        if hi - lo < 5:
            continue
        chunk = X[lo:hi, cols]
        stds = np.std(chunk, axis=0)
        active = stds > 1e-10
        if np.sum(active) < 2:
            continue
        C = np.corrcoef(chunk[:, active].T)
        C = np.nan_to_num(C, nan=0.0)
        triu = np.triu_indices(C.shape[0], k=1)
        result[i] = float(np.mean(np.abs(C[triu])))
    return result


def _rate_covariance(X, win):
    T, n_neurons = X.shape
    if n_neurons < 2:
        return np.zeros(T)
    half = win // 2
    result = np.zeros(T)
    n_check = min(n_neurons, 30)
    cols = np.linspace(0, n_neurons - 1, n_check, dtype=int)
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        if hi - lo < 5:
            continue
        chunk = X[lo:hi, cols]
        try:
            cov = np.cov(chunk.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            total = np.sum(np.abs(eigenvalues)) + 1e-10
            result[i] = float(eigenvalues[-1] / total)
        except Exception:
            pass
    return result


def compute_bio_variables(X_trial, Y_trial, epoch_mask, trial_meta, dt_ms=10):
    T = X_trial.shape[0]
    fs = 1000.0 / dt_ms
    variables = {}

    variables['firing_rate_input'] = np.mean(X_trial, axis=1)

    win = max(1, int(200 / dt_ms))
    fr_all = np.concatenate([X_trial, Y_trial], axis=1)
    mean_fr = np.mean(fr_all, axis=1)
    variables['trial_variance'] = _rolling_var(mean_fr, win)

    pop_rate = np.mean(X_trial, axis=1)
    variables['theta_power'] = _bandpower(pop_rate, fs, 4, 8)
    variables['gamma_power'] = _bandpower(pop_rate, fs, 30, min(80, fs / 2 - 1))
    variables['theta_gamma_pac'] = _pac_strength(pop_rate, fs)
    variables['population_synchrony'] = _population_synchrony(X_trial, win)
    variables['rate_covariance'] = _rate_covariance(X_trial, win)

    var_names = list(variables.keys())
    bio_targets = np.column_stack([variables[v] for v in var_names])
    return bio_targets, var_names


# =========================================================================
# TRAINING (identical to v2)
# =========================================================================

def gap_cv_split(n_trials, train_frac=0.7, gap_frac=0.1):
    n_train = int(n_trials * train_frac)
    n_gap = int(n_trials * gap_frac)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train + n_gap, n_trials))
    return train_idx, test_idx


def trials_to_npz(trials):
    X_data, Y_data = {}, {}
    for i, t in enumerate(trials):
        X_data['trial_{}'.format(i)] = t['X'].astype(np.float32)
        Y_data['trial_{}'.format(i)] = t['Y'].astype(np.float32)
    return X_data, Y_data


def collate_trials(X_data, Y_data, indices, device='cpu'):
    X_list = [torch.tensor(X_data['trial_{}'.format(i)], dtype=torch.float32) for i in indices]
    Y_list = [torch.tensor(Y_data['trial_{}'.format(i)], dtype=torch.float32) for i in indices]
    lengths = torch.tensor([x.shape[0] for x in X_list])
    X_pad = pad_sequence(X_list, batch_first=True).to(device)
    Y_pad = pad_sequence(Y_list, batch_first=True).to(device)
    return X_pad, Y_pad, lengths


def train_model(model, X_data, Y_data, train_idx, test_idx,
                max_epochs=300, patience=20, lr=1e-3, device='cpu'):
    X_tr, Y_tr, L_tr = collate_trials(X_data, Y_data, train_idx, device)
    X_te, Y_te, L_te = collate_trials(X_data, Y_data, test_idx, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_vl, patience_cnt, best_state = float('inf'), 0, None

    for ep in range(max_epochs):
        model.train()
        pred, _ = model(X_tr, L_tr)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for i, l in enumerate(L_tr):
            mask[i, :l] = True
        loss = crit(pred[mask], Y_tr[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_te, _ = model(X_te, L_te)
            mask_te = torch.zeros_like(pred_te, dtype=torch.bool)
            for i, l in enumerate(L_te):
                mask_te[i, :l] = True
            vl = crit(pred_te[mask_te], Y_te[mask_te]).item()

        if vl < best_vl:
            best_vl = vl
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_te, _ = model(X_te, L_te)
    all_pred, all_true = [], []
    for i, l in enumerate(L_te):
        all_pred.append(pred_te[i, :l].cpu().numpy())
        all_true.append(Y_te[i, :l].cpu().numpy())
    pred_flat = np.concatenate(all_pred).ravel()
    true_flat = np.concatenate(all_true).ravel()

    cc = 0.0
    if np.std(pred_flat) > 1e-10 and np.std(true_flat) > 1e-10:
        cc = float(np.corrcoef(pred_flat, true_flat)[0, 1])

    return model, cc, ep + 1


def extract_hidden_states(model, X_data, n_trials, device='cpu'):
    model.eval()
    hidden_dict = {}
    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key not in X_data:
            continue
        x = torch.tensor(X_data[key], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, h = model(x)
        hidden_dict[key] = h[0].cpu().numpy()
    return hidden_dict


# =========================================================================
# iAAFT SCREENING (unified dual gate — replaces delta-R-squared)
# =========================================================================

def iaaft_surrogate(signal, n_iterations=100, rng=None):
    """Generate iAAFT surrogate preserving power spectrum + amplitude distribution.

    Identical to the implementation in run_kyzar_phase3_4.py for consistency.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(signal)
    fft_orig = np.fft.rfft(signal)
    amplitudes = np.abs(fft_orig)
    sorted_original = np.sort(signal)

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


def iaaft_screen(H_trained, target, groups, n_surrogates=50,
                 alpha=1.0, r2_floor=0.01, random_state=42):
    """Unified iAAFT dual-gate screening.

    Gate 1: R2_trained > 95th percentile of iAAFT null (p < 0.05)
    Gate 2: R2_trained > r2_floor (effect size floor)

    Returns dict with screening results and diagnostics.
    """
    rng = np.random.default_rng(random_state)

    _fail = {
        'r2_trained': 0.0, 'iaaft_95th': 0.0, 'p_value': 1.0,
        'r2_floor': r2_floor, 'passes_screen': False, 'null_r2s': [],
    }

    if np.std(target) < 1e-10:
        return _fail

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        return _fail

    gkf = GroupKFold(n_splits=n_splits)

    # Observed R2_trained
    r2_trained = float(np.mean(cross_val_score(
        Ridge(alpha), H_trained, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

    # iAAFT null distribution
    null_r2s = []
    for si in range(n_surrogates):
        surr_target = iaaft_surrogate(target, rng=rng)
        if np.std(surr_target) < 1e-10:
            null_r2s.append(0.0)
            continue
        sr2 = float(np.mean(cross_val_score(
            Ridge(alpha), H_trained, surr_target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))
        null_r2s.append(sr2)

    null_arr = np.array(null_r2s)
    threshold_95 = float(np.percentile(null_arr, 95))
    p_value = float(np.mean(null_arr >= r2_trained))

    passes_iaaft = r2_trained > threshold_95
    passes_floor = r2_trained > r2_floor
    passes_screen = passes_iaaft and passes_floor

    return {
        'r2_trained': r2_trained,
        'iaaft_95th': threshold_95,
        'p_value': p_value,
        'r2_floor': r2_floor,
        'passes_iaaft': passes_iaaft,
        'passes_floor': passes_floor,
        'passes_screen': passes_screen,
        'null_r2s': null_r2s,
    }


# =========================================================================
# RESAMPLE ABLATION (identical to v2)
# =========================================================================

def resample_ablation(H_trained, target, groups, n_resamples=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    hdim = H_trained.shape[1]
    k_values = sorted(set(max(1, int(hdim * f)) for f in [0.10, 0.20, 0.30, 0.40]))
    gkf = GroupKFold(n_splits=5)

    if np.std(target) < 1e-10:
        return {'baseline_r2': 0.0, 'per_k': {}, 'overall_verdict': 'BYPRODUCT'}

    baseline_r2 = float(np.mean(cross_val_score(
        Ridge(1.0), H_trained, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

    dim_corrs = np.array([abs(np.corrcoef(H_trained[:, d], target)[0, 1])
                          if np.std(H_trained[:, d]) > 1e-10 else 0.0
                          for d in range(hdim)])

    results_per_k = {}
    for k in k_values:
        top_k = np.argsort(dim_corrs)[-k:]
        H_ablated = H_trained.copy()
        H_ablated[:, top_k] = 0.0
        ablated_r2 = float(np.mean(cross_val_score(
            Ridge(1.0), H_ablated, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))

        rand_r2s = []
        for _ in range(n_resamples):
            rd = rng.choice(hdim, k, replace=False)
            H_r = H_trained.copy()
            H_r[:, rd] = 0.0
            rr2 = float(np.mean(cross_val_score(
                Ridge(1.0), H_r, target, cv=gkf, groups=groups, scoring='r2', n_jobs=-1)))
            rand_r2s.append(rr2)

        rand_mean = float(np.mean(rand_r2s))
        rand_std = float(np.std(rand_r2s)) + 1e-10
        z = (ablated_r2 - rand_mean) / rand_std

        results_per_k[k] = {
            'k': k, 'baseline_r2': baseline_r2,
            'ablated_r2': ablated_r2, 'degradation': baseline_r2 - ablated_r2,
            'rand_mean': rand_mean, 'rand_std': rand_std,
            'z_score': z, 'verdict': 'CAUSAL' if z < -2.0 else 'BYPRODUCT',
        }

    n_causal = sum(1 for r in results_per_k.values() if r['verdict'] == 'CAUSAL')
    return {
        'baseline_r2': baseline_r2, 'per_k': results_per_k,
        'n_causal_k': n_causal,
        'overall_verdict': 'MANDATORY' if n_causal >= 2 else
                           'CAUSAL' if n_causal >= 1 else 'BYPRODUCT',
    }


# =========================================================================
# FULL CONDITION PIPELINE (v3: iAAFT screening replaces delta-R-squared)
# =========================================================================

def run_condition(condition_name, trials, original_trials, hidden_dim,
                  device, rng, output_dir, n_surrogates=50):
    """Full pipeline: verify -> train -> iAAFT screen -> ablate survivors."""
    log.info("\n" + "=" * 70)
    log.info("CONDITION: %s", condition_name)
    log.info("=" * 70)

    # --- VERIFY TRANSFORM ---
    verification = verify_transform(original_trials, trials, condition_name)
    if not verification['valid']:
        log.warning("  *** VERIFICATION FAILED for %s ***", condition_name)

    n_trials = len(trials)
    n_in = trials[0]['X'].shape[1]
    n_out = trials[0]['Y'].shape[1]

    # --- TRAIN ---
    X_data, Y_data = trials_to_npz(trials)
    train_idx, test_idx = gap_cv_split(n_trials)

    log.info("  Training LSTM (h=%d) on %d trials...", hidden_dim, n_trials)
    torch.manual_seed(42)
    model = LSTMSurrogate(n_in, n_out, hidden_dim).to(device)
    model, cc, n_epochs = train_model(
        model, X_data, Y_data, train_idx, test_idx, device=device)
    log.info("  CC=%.3f (%d epochs)", cc, n_epochs)

    # --- EXTRACT HIDDEN STATES (trained only -- no untrained needed for iAAFT) ---
    h_trained = extract_hidden_states(model, X_data, n_trials, device)

    # --- COMPUTE BIO VARIABLES ---
    log.info("  Computing 7 biological variables...")
    bio_all, trial_groups_all = [], []
    h_tr_all = []
    bio_names = None

    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key not in h_trained:
            continue
        bio, names = compute_bio_variables(
            trials[ti]['X'], trials[ti]['Y'],
            trials[ti]['epoch'], trials[ti].get('meta', {}))
        if bio_names is None:
            bio_names = names
        ht = h_trained[key]
        min_len = min(len(ht), len(bio))
        h_tr_all.append(ht[:min_len])
        bio_all.append(bio[:min_len])
        trial_groups_all.extend([ti] * min_len)

    H_t = np.concatenate(h_tr_all, axis=0).astype(np.float64)
    bio_targets = np.nan_to_num(np.concatenate(bio_all, axis=0).astype(np.float64), nan=0.0)
    groups = np.array(trial_groups_all)

    log.info("  Samples: %d, Groups: %d unique trials", len(H_t), len(np.unique(groups)))

    # =====================================================================
    # v3 CHANGE: iAAFT dual-gate screening replaces delta-R-squared
    # =====================================================================
    log.info("  iAAFT screening (%d surrogates, R2 floor=0.01)...", n_surrogates)
    screen_results = {}
    for vi, vname in enumerate(bio_names):
        target = bio_targets[:, vi]
        result = iaaft_screen(H_t, target, groups,
                              n_surrogates=n_surrogates, random_state=42 + vi)
        screen_results[vname] = result

        status = "PASS" if result['passes_screen'] else "FAIL"
        reason = ""
        if not result['passes_floor']:
            reason = " (R2<0.01)"
        elif not result['passes_iaaft']:
            reason = " (p>=0.05)"
        log.info("    %-25s R2=%.4f  95th=%.4f  p=%.3f  [%s%s]",
                 vname, result['r2_trained'], result['iaaft_95th'],
                 result['p_value'], status, reason)

    n_pass = sum(1 for r in screen_results.values() if r['passes_screen'])
    log.info("  iAAFT screening: %d/%d passed dual gate", n_pass, len(bio_names))

    # --- RESAMPLE ABLATION on survivors ---
    log.info("  Resample ablation on %d survivors...", n_pass)
    ablation_results = {}
    abl_rng = np.random.default_rng(42)
    for vi, vname in enumerate(bio_names):
        if screen_results[vname]['passes_screen']:
            target = bio_targets[:, vi]
            abl = resample_ablation(H_t, target, groups, rng=abl_rng)
            ablation_results[vname] = abl
            log.info("    %-25s %s (z-scores: %s)",
                     vname, abl['overall_verdict'],
                     ', '.join('{:.1f}'.format(r['z_score'])
                               for r in abl['per_k'].values()))
        else:
            ablation_results[vname] = {
                'overall_verdict': 'NOT_TESTED',
                'reason': 'Failed iAAFT screening',
            }

    # --- COMPILE ---
    mandatory_vars = [v for v, r in ablation_results.items()
                      if r.get('overall_verdict') == 'MANDATORY']
    n_mandatory = len(mandatory_vars)
    n_causal = sum(1 for v in ablation_results.values()
                   if v.get('overall_verdict') in ('MANDATORY', 'CAUSAL'))

    log.info("\n  --- %s SUMMARY ---", condition_name)
    log.info("  CC: %.3f", cc)
    log.info("  iAAFT passed: %d/7", n_pass)
    log.info("  Mandatory: %d/7 -- %s", n_mandatory,
             ', '.join(mandatory_vars) if mandatory_vars else 'NONE (ZOMBIE)')
    log.info("  Causal: %d/7", n_causal)
    log.info("  Verification: %s", "PASS" if verification['valid'] else "FAIL")

    result = {
        'condition': condition_name,
        'output_cc': cc,
        'n_epochs': n_epochs,
        'n_trials': n_trials,
        'n_samples': len(H_t),
        'hidden_dim': hidden_dim,
        'screening_method': 'iAAFT_dual_gate',
        'n_surrogates': n_surrogates,
        'verification': verification,
        'iaaft_screening': {v: {k: _convert_numpy(val) for k, val in r.items() if k != 'null_r2s'}
                           for v, r in screen_results.items()},
        'ablation': ablation_results,
        'summary': {
            'n_iaaft_passed': n_pass,
            'n_mandatory': n_mandatory,
            'n_causal': n_causal,
            'mandatory_variables': mandatory_vars,
            'verification_valid': verification['valid'],
        },
    }

    cond_dir = Path(output_dir) / condition_name.lower()
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / 'results.json', 'w') as f:
        json.dump(_convert_numpy(result), f, indent=2)

    return result


# =========================================================================
# COMPARISON TABLE
# =========================================================================

def print_comparison_table(results):
    log.info("\n" + "=" * 90)
    log.info("SYNTHETIC REALITY v3 (iAAFT) -- COMPARISON TABLE")
    log.info("Screening: R2_trained > iAAFT 95th AND R2_trained > 0.01")
    log.info("=" * 90)

    header = '{:<22}'.format('Variable')
    for r in results:
        header += ' | {:>12}'.format(r['condition'][:12])
    log.info(header)
    log.info("-" * len(header))

    for vname in PROBE_VARIABLES:
        line = '{:<22}'.format(vname)
        for r in results:
            scr = r['iaaft_screening'].get(vname, {})
            r2 = scr.get('r2_trained', 0)
            abl = r['ablation'].get(vname, {})
            verdict = abl.get('overall_verdict', '-')
            if verdict == 'MANDATORY':
                marker = ' ***'
            elif verdict == 'CAUSAL':
                marker = '  + '
            elif verdict == 'BYPRODUCT':
                marker = '    '
            else:
                marker = '  - '
            line += ' | {:>7.4f}{}'.format(r2, marker)
        log.info(line)

    log.info("-" * len(header))
    log.info("  *** = MANDATORY   + = CAUSAL   - = failed screen   (blank) = BYPRODUCT")

    log.info("")
    for r in results:
        mvars = r['summary']['mandatory_variables']
        n_pass = r['summary']['n_iaaft_passed']
        if mvars:
            log.info("  %s: %d/7 screened, MANDATORY = %s",
                     r['condition'], n_pass, ', '.join(mvars))
        else:
            log.info("  %s: %d/7 screened, ZOMBIE (0 mandatory)", r['condition'], n_pass)


def print_divergence_analysis(results):
    """Check whether conditions produce different mandatory sets."""
    log.info("\n" + "=" * 70)
    log.info("DIVERGENCE ANALYSIS")
    log.info("=" * 70)

    mandatory_sets = {}
    for r in results:
        mandatory_sets[r['condition']] = set(r['summary']['mandatory_variables'])

    # Check pairwise overlap
    conditions = list(mandatory_sets.keys())
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            s1, s2 = mandatory_sets[c1], mandatory_sets[c2]
            shared = s1 & s2
            only_1 = s1 - s2
            only_2 = s2 - s1
            log.info("")
            log.info("  %s vs %s:", c1, c2)
            log.info("    Shared:    %s", ', '.join(sorted(shared)) if shared else 'NONE')
            log.info("    Only %s: %s", c1[:8], ', '.join(sorted(only_1)) if only_1 else 'NONE')
            log.info("    Only %s: %s", c2[:8], ', '.join(sorted(only_2)) if only_2 else 'NONE')

    # Key question: are all sets identical?
    all_sets = list(mandatory_sets.values())
    all_identical = all(s == all_sets[0] for s in all_sets)

    log.info("")
    if all_identical and len(all_sets[0]) > 0:
        log.info("  *** ALL CONDITIONS PRODUCE IDENTICAL MANDATORY SETS ***")
        log.info("  *** These variables are ABSOLUTE ARCHITECTURAL NECESSITIES ***")
        log.info("  *** NOT reality-dependent reflections ***")
    elif all_identical and len(all_sets[0]) == 0:
        log.info("  *** ALL CONDITIONS ARE ZOMBIES ***")
    else:
        log.info("  *** CONDITIONS PRODUCE DIFFERENT MANDATORY SETS ***")
        log.info("  *** Mandatory variables are REALITY-DEPENDENT ***")

    # Specific predictions
    seq_mand = mandatory_sets.get('SEQUENTIAL', set())
    spa_mand = mandatory_sets.get('SPATIAL_ONLY', set())
    noi_mand = mandatory_sets.get('PURE_NOISE', set())

    log.info("")
    log.info("  PREDICTION CHECK:")
    log.info("    SEQUENTIAL -> theta_gamma_pac mandatory?  %s",
             "YES" if 'theta_gamma_pac' in seq_mand else "NO")
    log.info("    SPATIAL_ONLY -> gamma_power mandatory?    %s",
             "YES" if 'gamma_power' in spa_mand else "NO")
    log.info("    PURE_NOISE -> 0 mandatory (zombie)?       %s",
             "YES" if len(noi_mand) == 0 else "NO (%d mandatory)" % len(noi_mand))


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Synthetic Reality v3 -- iAAFT Screening')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--output-dir', default='results/synthetic_reality_v3')
    parser.add_argument('--source-subject', default='5')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-surrogates', type=int, default=50,
                        help='Number of iAAFT surrogates per variable (default: 50)')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    log.info("=" * 70)
    log.info("SYNTHETIC REALITY v3 -- iAAFT UNIFIED SCREENING")
    log.info("  Screening: R2_trained > iAAFT 95th AND R2 > 0.01 (replaces delta-R2)")
    log.info("  Source subject: sub-%s", args.source_subject)
    log.info("  Hidden dim: %d", args.hidden_dim)
    log.info("  Device: %s", args.device)
    log.info("  iAAFT surrogates: %d", args.n_surrogates)
    log.info("  Conditions: %s", ', '.join(VERIFIED_CONDITIONS))
    log.info("  (RATE_MODULATED, TEMPORAL_ONLY excluded: failed v2 verification)")
    log.info("  Probe variables: %s", ', '.join(PROBE_VARIABLES))
    log.info("=" * 70)

    t_start = time.time()

    # Load source data
    trials, meta = load_source_data(args.processed_dir, args.source_subject)

    # Generate only the 3 verified conditions
    log.info("\n--- GENERATING CONDITIONS ---")
    cond_seq = condition_sequential(trials)
    cond_spa = condition_spatial_only(trials, rng)
    cond_noi = condition_pure_noise(trials, rng)

    conditions = [
        ('SEQUENTIAL', cond_seq),
        ('SPATIAL_ONLY', cond_spa),
        ('PURE_NOISE', cond_noi),
    ]

    # Run each condition
    all_results = []
    for cond_name, cond_trials in conditions:
        r = run_condition(cond_name, cond_trials, trials,
                          args.hidden_dim, args.device, rng,
                          str(output_dir), n_surrogates=args.n_surrogates)
        all_results.append(r)

    # Comparison and analysis
    print_comparison_table(all_results)
    print_divergence_analysis(all_results)

    # Save master results
    total_time = time.time() - t_start
    master = {
        'experiment': 'synthetic_reality_v3_iaaft',
        'screening_method': 'iAAFT dual gate (R2>95th AND R2>0.01)',
        'v2_change': 'Replaced delta-R2 screening with unified iAAFT. Excluded RATE_MODULATED and TEMPORAL_ONLY (failed v2 verification).',
        'source_subject': args.source_subject,
        'hidden_dim': args.hidden_dim,
        'n_surrogates': args.n_surrogates,
        'total_time_min': total_time / 60,
        'conditions': [_convert_numpy(r) for r in all_results],
        'predictions': {
            'sequential': 'theta_gamma_pac mandatory (temporal structure)',
            'spatial_only': 'gamma_power mandatory (spatial co-activation)',
            'pure_noise': '0 mandatory (zombie)',
            'key_test': 'If all identical -> absolute necessities. If different -> reality-dependent.',
        },
    }
    with open(output_dir / 'synthetic_reality_v3_results.json', 'w') as f:
        json.dump(_convert_numpy(master), f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("SYNTHETIC REALITY v3 COMPLETE (%.1f min)", total_time / 60)
    log.info("Results: %s", output_dir)
    log.info("=" * 70)


if __name__ == '__main__':
    main()
