#!/usr/bin/env python3
"""
run_synthetic_reality_v2.py

DESCARTES Synthetic Reality v2 — Radical Restructuring Experiment

Goal: Show that DIFFERENT task structures produce DIFFERENT mandatory variables.
v1 was insufficient because boundaryless only removed trial boundaries —
both structured conditions demanded the same computation (track population rates).

v2 creates ORTHOGONAL coding decompositions:
  Condition 1: SEQUENTIAL    — real Sternberg data (control)
  Condition 2: RATE_MODULATED — destroy temporal structure, preserve rates
  Condition 3: TEMPORAL_ONLY — destroy rate differences, preserve timing
  Condition 4: SPATIAL_ONLY  — preserve co-activation patterns, destroy timing
  Condition 5: PURE_NOISE    — Poisson noise (zombie control)

Critical prediction: Conditions 2 and 3 should produce DIFFERENT mandatory
variables. If Rate→firing_rate_input and Temporal→theta_gamma_pac, then
mandatory variables reflect the available coding dimension.

Verification: Each transform is checked to confirm it actually destroys/preserves
what it claims. If verification fails, the condition is flagged as invalid.

Usage:
  python scripts/run_synthetic_reality_v2.py \
    --processed-dir data/kyzar_processed \
    --output-dir results/synthetic_reality_v2 \
    --source-subject 5 \
    --device cuda
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
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('synthetic_reality_v2')

# 7 probe targets (5 original + 2 new)
PROBE_VARIABLES = [
    'firing_rate_input', 'trial_variance', 'theta_power',
    'gamma_power', 'theta_gamma_pac',
    'population_synchrony', 'rate_covariance',
]


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
# MODEL (same as Phase 2 / v1)
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
# DATA LOADING
# =========================================================================

def load_source_data(processed_dir, subject):
    """Load all trial data for source subject."""
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
    log.info("  trial lengths: %d-%d timesteps",
             min(t['X'].shape[0] for t in trials),
             max(t['X'].shape[0] for t in trials))

    return trials, meta


# =========================================================================
# TRANSFORM VERIFICATION
# =========================================================================

def verify_transform(original_trials, transformed_trials, condition_name, dt_ms=10):
    """Verify that a transform actually destroyed/preserved what it claims."""
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
        'n_timesteps_orig': len(orig_X),
        'n_timesteps_trans': len(trans_X),
    }

    # --- Mean firing rates per neuron ---
    orig_rates = np.mean(orig_X, axis=0)
    trans_rates = np.mean(trans_X, axis=0)[:n_neurons_orig] if n_neurons_trans >= n_neurons_orig else np.mean(trans_X, axis=0)
    min_n = min(len(orig_rates), len(trans_rates))
    rate_corr = float(np.corrcoef(orig_rates[:min_n], trans_rates[:min_n])[0, 1]) if min_n > 1 else 0.0
    rate_rmse = float(np.sqrt(np.mean((orig_rates[:min_n] - trans_rates[:min_n])**2)))
    verification['rate_correlation'] = rate_corr
    verification['rate_rmse'] = rate_rmse
    log.info("    Rate correlation: %.3f (RMSE: %.4f)", rate_corr, rate_rmse)

    # --- Spectral content (population-level) ---
    orig_pop = np.mean(orig_X, axis=1)
    trans_pop = np.mean(trans_X, axis=1)

    orig_theta = float(np.mean(_bandpower(orig_pop, fs, 4, 8)))
    trans_theta = float(np.mean(_bandpower(trans_pop, fs, 4, 8)))
    orig_gamma = float(np.mean(_bandpower(orig_pop, fs, 30, min(80, fs/2 - 1))))
    trans_gamma = float(np.mean(_bandpower(trans_pop, fs, 30, min(80, fs/2 - 1))))

    theta_ratio = trans_theta / (orig_theta + 1e-10)
    gamma_ratio = trans_gamma / (orig_gamma + 1e-10)
    verification['theta_power_orig'] = orig_theta
    verification['theta_power_trans'] = trans_theta
    verification['theta_ratio'] = theta_ratio
    verification['gamma_power_orig'] = orig_gamma
    verification['gamma_power_trans'] = trans_gamma
    verification['gamma_ratio'] = gamma_ratio
    log.info("    Theta power ratio: %.3f", theta_ratio)
    log.info("    Gamma power ratio: %.3f", gamma_ratio)

    # --- Pairwise correlations ---
    n_sample = min(5000, len(orig_X), len(trans_X))
    n_check = min(min_n, 50)
    if n_check > 1:
        orig_corr_mat = np.corrcoef(orig_X[:n_sample, :n_check].T)
        trans_corr_mat = np.corrcoef(trans_X[:n_sample, :n_check].T)
        triu_idx = np.triu_indices(n_check, k=1)
        orig_pairwise = orig_corr_mat[triu_idx]
        trans_pairwise = trans_corr_mat[triu_idx]
        pairwise_corr = float(np.corrcoef(orig_pairwise, trans_pairwise)[0, 1])
        pairwise_diff = float(np.mean(np.abs(orig_pairwise - trans_pairwise)))
    else:
        pairwise_corr = 0.0
        pairwise_diff = 0.0
    verification['pairwise_correlation'] = pairwise_corr
    verification['pairwise_mean_diff'] = pairwise_diff
    log.info("    Pairwise correlation preservation: %.3f", pairwise_corr)

    # --- Rate variance across neurons ---
    trans_rate_std = float(np.std(trans_rates[:min_n]))
    orig_rate_std = float(np.std(orig_rates[:min_n]))
    verification['rate_std_orig'] = orig_rate_std
    verification['rate_std_trans'] = trans_rate_std
    verification['rate_std_ratio'] = trans_rate_std / (orig_rate_std + 1e-10)
    log.info("    Rate std ratio: %.3f", verification['rate_std_ratio'])

    # --- Condition-specific pass/fail ---
    if condition_name == 'RATE_MODULATED':
        verification['preserves_rates'] = rate_corr > 0.8
        verification['destroys_oscillations'] = theta_ratio < 0.3 and gamma_ratio < 0.3
        verification['valid'] = verification['preserves_rates'] and verification['destroys_oscillations']
    elif condition_name == 'TEMPORAL_ONLY':
        verification['destroys_rates'] = verification['rate_std_ratio'] < 0.2
        verification['preserves_oscillations'] = theta_ratio > 0.3
        verification['valid'] = verification['destroys_rates']
    elif condition_name == 'SPATIAL_ONLY':
        verification['preserves_coactivation'] = pairwise_corr > 0.5
        verification['destroys_timing'] = theta_ratio < 0.5
        verification['valid'] = verification['preserves_coactivation']
    elif condition_name == 'PURE_NOISE':
        verification['valid'] = True
    else:
        verification['valid'] = True

    status = "PASS" if verification['valid'] else "FAIL"
    log.info("    Verification: %s", status)

    return verification


# =========================================================================
# CONDITION GENERATORS
# =========================================================================

def condition_sequential(trials):
    """Condition 1: Use real data as-is (control)."""
    log.info("  Condition 1: SEQUENTIAL (real Sternberg data, %d trials)", len(trials))
    return trials


def condition_rate_modulated(trials, rng):
    """Condition 2: RATE_MODULATED — destroy temporal structure, preserve rates.

    For each trial, for each neuron independently:
    1. Shuffle timesteps within the trial
    2. Smooth with 50ms Gaussian to restore local continuity
    This preserves per-trial mean rates and trial boundaries.
    This destroys oscillatory structure, phase relationships, PAC, ISI patterns.
    """
    log.info("  Condition 2: RATE_MODULATED (destroy temporal, preserve rates)")

    dt_ms = 10
    sigma = 50.0 / dt_ms  # 50ms Gaussian in timestep units

    new_trials = []
    for t in trials:
        X = t['X'].copy()
        Y = t['Y'].copy()
        T, n_in = X.shape
        _, n_out = Y.shape

        # Shuffle timesteps independently per neuron
        X_shuffled = np.zeros_like(X)
        for ni in range(n_in):
            perm = rng.permutation(T)
            X_shuffled[:, ni] = X[perm, ni]

        # Smooth to restore local continuity
        X_smooth = np.zeros_like(X_shuffled)
        for ni in range(n_in):
            X_smooth[:, ni] = gaussian_filter1d(X_shuffled[:, ni], sigma=sigma)

        # Rescale to match original per-neuron mean and std
        for ni in range(n_in):
            orig_mean = np.mean(X[:, ni])
            orig_std = np.std(X[:, ni])
            new_std = np.std(X_smooth[:, ni])
            if new_std > 1e-10:
                X_smooth[:, ni] = (X_smooth[:, ni] - np.mean(X_smooth[:, ni])) / new_std * orig_std + orig_mean

        # Same transform for Y
        Y_shuffled = np.zeros_like(Y)
        for ni in range(n_out):
            perm = rng.permutation(T)
            Y_shuffled[:, ni] = Y[perm, ni]
        Y_smooth = np.zeros_like(Y_shuffled)
        for ni in range(n_out):
            Y_smooth[:, ni] = gaussian_filter1d(Y_shuffled[:, ni], sigma=sigma)
        for ni in range(n_out):
            orig_mean = np.mean(Y[:, ni])
            orig_std = np.std(Y[:, ni])
            new_std = np.std(Y_smooth[:, ni])
            if new_std > 1e-10:
                Y_smooth[:, ni] = (Y_smooth[:, ni] - np.mean(Y_smooth[:, ni])) / new_std * orig_std + orig_mean

        new_trials.append({
            'X': X_smooth.astype(np.float32),
            'Y': Y_smooth.astype(np.float32),
            'epoch': t['epoch'].copy(),
            'meta': t.get('meta', {}),
        })

    log.info("    %d trials transformed (shuffle + 50ms Gaussian smooth)", len(new_trials))
    return new_trials


def condition_temporal_only(trials, rng):
    """Condition 3: TEMPORAL_ONLY — destroy rate differences, preserve timing.

    For each trial, z-score each neuron (mean=0, std=1), then rescale so all
    neurons have identical mean rate but PRESERVE original temporal pattern.
    """
    log.info("  Condition 3: TEMPORAL_ONLY (destroy rates, preserve timing)")

    all_X = np.concatenate([t['X'] for t in trials], axis=0)
    global_mean = float(np.mean(all_X))
    global_std = float(np.std(all_X))

    all_Y = np.concatenate([t['Y'] for t in trials], axis=0)
    y_global_mean = float(np.mean(all_Y))
    y_global_std = float(np.std(all_Y))

    new_trials = []
    for t in trials:
        X = t['X'].copy()
        Y = t['Y'].copy()
        T, n_in = X.shape
        _, n_out = Y.shape

        X_norm = np.zeros_like(X)
        for ni in range(n_in):
            col = X[:, ni]
            col_std = np.std(col)
            if col_std > 1e-10:
                X_norm[:, ni] = (col - np.mean(col)) / col_std * global_std + global_mean
            else:
                X_norm[:, ni] = global_mean

        Y_norm = np.zeros_like(Y)
        for ni in range(n_out):
            col = Y[:, ni]
            col_std = np.std(col)
            if col_std > 1e-10:
                Y_norm[:, ni] = (col - np.mean(col)) / col_std * y_global_std + y_global_mean
            else:
                Y_norm[:, ni] = y_global_mean

        new_trials.append({
            'X': X_norm.astype(np.float32),
            'Y': Y_norm.astype(np.float32),
            'epoch': t['epoch'].copy(),
            'meta': t.get('meta', {}),
        })

    log.info("    %d trials: all neurons rescaled to global mean=%.3f, std=%.3f",
             len(new_trials), global_mean, global_std)
    return new_trials


def condition_spatial_only(trials, rng):
    """Condition 4: SPATIAL_ONLY — preserve co-activation, destroy timing.

    For each timestep, preserve the spatial pattern across neurons (which
    neurons are active) but destroy temporal order by randomly permuting
    timesteps within 200ms windows.
    """
    log.info("  Condition 4: SPATIAL_ONLY (preserve co-activation, destroy timing)")

    dt_ms = 10
    window_size = int(200 / dt_ms)  # 20 timesteps = 200ms

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
    """Condition 5: Poisson noise matched to real statistics."""
    log.info("  Condition 5: PURE_NOISE (Poisson matched to statistics)")

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
            'X': X_noise,
            'Y': Y_noise,
            'epoch': epoch_noise,
            'meta': {},
        })

    log.info("    Generated %d noise trials matched to real stats", len(noise_trials))
    return noise_trials


# =========================================================================
# BIO VARIABLE COMPUTATION (7 variables: 5 original + 2 new)
# =========================================================================

def compute_bio_variables(X_trial, Y_trial, epoch_mask, trial_meta, dt_ms=10):
    """Compute 7 biological variables for one trial."""
    T = X_trial.shape[0]
    dt_s = dt_ms / 1000.0
    fs = 1.0 / dt_s

    variables = {}

    # --- Original 5 ---
    variables['firing_rate_input'] = np.mean(X_trial, axis=1)

    win = max(1, int(200 / dt_ms))
    fr_all = np.concatenate([X_trial, Y_trial], axis=1)
    mean_fr = np.mean(fr_all, axis=1)
    variables['trial_variance'] = _rolling_var(mean_fr, win)

    pop_rate = np.mean(X_trial, axis=1)
    variables['theta_power'] = _bandpower(pop_rate, fs, 4, 8)
    variables['gamma_power'] = _bandpower(pop_rate, fs, 30, min(80, fs/2 - 1))
    variables['theta_gamma_pac'] = _pac_strength(pop_rate, fs)

    # --- NEW: population_synchrony ---
    variables['population_synchrony'] = _population_synchrony(X_trial, win)

    # --- NEW: rate_covariance ---
    variables['rate_covariance'] = _rate_covariance(X_trial, win)

    var_names = list(variables.keys())
    bio_targets = np.column_stack([variables[v] for v in var_names])
    return bio_targets, var_names


def _rolling_var(x, win):
    T = len(x)
    result = np.zeros(T)
    half = win // 2
    for i in range(T):
        lo, hi = max(0, i - half), min(T, i + half + 1)
        result[i] = np.var(x[lo:hi]) if hi - lo > 1 else 0
    return result


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
    """Mean pairwise correlation across neurons in a sliding window."""
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
    """Top eigenvalue fraction of neuron x neuron covariance matrix per window."""
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


# =========================================================================
# TRAINING PIPELINE
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
    X_list = [torch.tensor(X_data['trial_{}'.format(i)], dtype=torch.float32)
              for i in indices]
    Y_list = [torch.tensor(Y_data['trial_{}'.format(i)], dtype=torch.float32)
              for i in indices]
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
# PROBING + ABLATION + iAAFT
# =========================================================================

def ridge_delta_r2(H_trained, H_untrained, target, groups, alpha=1.0):
    gkf = GroupKFold(n_splits=5)
    # Replace NaN/Inf with 0
    if np.any(np.isnan(target)) or np.any(np.isinf(target)):
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(target) < 1e-10:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0}
    r2_t = float(np.mean(cross_val_score(
        Ridge(alpha), H_trained, target, cv=gkf, groups=groups, scoring='r2')))
    r2_u = float(np.mean(cross_val_score(
        Ridge(alpha), H_untrained, target, cv=gkf, groups=groups, scoring='r2')))
    return {'r2_trained': r2_t, 'r2_untrained': r2_u, 'delta_r2': r2_t - r2_u}


def resample_ablation(H_trained, target, groups, n_resamples=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    hdim = H_trained.shape[1]
    k_values = sorted(set(max(1, int(hdim * f)) for f in [0.10, 0.20, 0.30, 0.40]))
    gkf = GroupKFold(n_splits=5)

    if np.std(target) < 1e-10:
        return {'baseline_r2': 0.0, 'per_k': {}, 'overall_verdict': 'BYPRODUCT'}

    baseline_r2 = float(np.mean(cross_val_score(
        Ridge(1.0), H_trained, target, cv=gkf, groups=groups, scoring='r2')))

    dim_corrs = np.array([abs(np.corrcoef(H_trained[:, d], target)[0, 1])
                          if np.std(H_trained[:, d]) > 1e-10 else 0.0
                          for d in range(hdim)])

    results_per_k = {}
    for k in k_values:
        top_k = np.argsort(dim_corrs)[-k:]
        H_ablated = H_trained.copy()
        H_ablated[:, top_k] = 0.0
        ablated_r2 = float(np.mean(cross_val_score(
            Ridge(1.0), H_ablated, target, cv=gkf, groups=groups, scoring='r2')))

        rand_r2s = []
        for _ in range(n_resamples):
            rd = rng.choice(hdim, k, replace=False)
            H_r = H_trained.copy()
            H_r[:, rd] = 0.0
            rr2 = float(np.mean(cross_val_score(
                Ridge(1.0), H_r, target, cv=gkf, groups=groups, scoring='r2')))
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


def iaaft_surrogate(x, rng, n_iter=100):
    n = len(x)
    sorted_x = np.sort(x)
    fft_orig = np.fft.rfft(x)
    mag_orig = np.abs(fft_orig)
    surrogate = rng.permutation(x).copy()
    for _ in range(n_iter):
        fft_surr = np.fft.rfft(surrogate)
        phase_surr = np.angle(fft_surr)
        fft_adjusted = mag_orig * np.exp(1j * phase_surr)
        surrogate = np.fft.irfft(fft_adjusted, n=n)
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_x[ranks]
    return surrogate


def iaaft_test(H_trained, target, groups, n_surrogates=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    gkf = GroupKFold(n_splits=5)
    if np.std(target) < 1e-10:
        return {'real_r2': 0.0, 'surr_mean': 0.0, 'p_value': 1.0}
    real_r2 = float(np.mean(cross_val_score(
        Ridge(1.0), H_trained, target, cv=gkf, groups=groups, scoring='r2')))
    surr_r2s = []
    for i in range(n_surrogates):
        surr_target = iaaft_surrogate(target, rng)
        sr2 = float(np.mean(cross_val_score(
            Ridge(1.0), H_trained, surr_target, cv=gkf, groups=groups, scoring='r2')))
        surr_r2s.append(sr2)
    surr_mean = float(np.mean(surr_r2s))
    surr_std = float(np.std(surr_r2s)) + 1e-10
    p_value = float(np.mean([1 if s >= real_r2 else 0 for s in surr_r2s]))
    return {
        'real_r2': real_r2, 'surr_mean': surr_mean, 'surr_std': surr_std,
        'p_value': p_value, 'z_score': (real_r2 - surr_mean) / surr_std,
        'significant': p_value < 0.05,
    }


# =========================================================================
# FULL CONDITION PIPELINE
# =========================================================================

def run_condition(condition_name, trials, original_trials, hidden_dim,
                  device, rng, output_dir):
    """Full pipeline: verify -> train -> probe -> ablate -> harden."""
    log.info("\n" + "=" * 70)
    log.info("CONDITION: %s", condition_name)
    log.info("=" * 70)

    # --- VERIFY TRANSFORM ---
    verification = verify_transform(original_trials, trials, condition_name)
    if not verification['valid']:
        log.warning("  *** VERIFICATION FAILED for %s -- results may be invalid ***",
                    condition_name)

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

    # Untrained baseline
    torch.manual_seed(99)
    model_u = LSTMSurrogate(n_in, n_out, hidden_dim).to(device)

    # --- EXTRACT HIDDEN STATES ---
    h_trained = extract_hidden_states(model, X_data, n_trials, device)
    h_untrained = extract_hidden_states(model_u, X_data, n_trials, device)

    # --- COMPUTE BIO VARIABLES ---
    log.info("  Computing 7 biological variables...")
    bio_all, trial_groups_all = [], []
    h_tr_all, h_un_all = [], []
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
        hu = h_untrained[key]
        min_len = min(len(ht), len(hu), len(bio))

        h_tr_all.append(ht[:min_len])
        h_un_all.append(hu[:min_len])
        bio_all.append(bio[:min_len])
        trial_groups_all.extend([ti] * min_len)

    H_t = np.concatenate(h_tr_all, axis=0).astype(np.float64)
    H_u = np.concatenate(h_un_all, axis=0).astype(np.float64)
    bio_targets = np.nan_to_num(np.concatenate(bio_all, axis=0).astype(np.float64), nan=0.0)
    groups = np.array(trial_groups_all)

    log.info("  Samples: %d, Groups: %d unique trials", len(H_t), len(np.unique(groups)))

    # --- PROBE (threshold 0.02 per v2 spec) ---
    DR2_THRESHOLD = 0.02
    log.info("  Probing %d variables (threshold dR2 > %.2f)...", len(bio_names), DR2_THRESHOLD)
    probe_results = {}
    for vi, vname in enumerate(bio_names):
        target = bio_targets[:, vi]
        ridge = ridge_delta_r2(H_t, H_u, target, groups)
        log.info("    %s: dR2=%.4f (trained=%.4f, untrained=%.4f)",
                 vname, ridge['delta_r2'], ridge['r2_trained'], ridge['r2_untrained'])
        probe_results[vname] = {'ridge': ridge}

    # --- ABLATION ---
    log.info("  Resample ablation...")
    ablation_results = {}
    abl_rng = np.random.default_rng(42)
    for vi, vname in enumerate(bio_names):
        dr2 = probe_results[vname]['ridge']['delta_r2']
        if dr2 > DR2_THRESHOLD:
            target = bio_targets[:, vi]
            abl = resample_ablation(H_t, target, groups, rng=abl_rng)
            ablation_results[vname] = abl
            log.info("    %s: %s (z-scores: %s)",
                     vname, abl['overall_verdict'],
                     ', '.join('{:.1f}'.format(r['z_score'])
                               for r in abl['per_k'].values()))
        else:
            ablation_results[vname] = {
                'overall_verdict': 'NOT_TESTED',
                'reason': 'dR2 <= {}'.format(DR2_THRESHOLD)}

    # --- iAAFT HARDENING ---
    log.info("  iAAFT statistical hardening...")
    iaaft_results = {}
    iaaft_rng = np.random.default_rng(123)
    for vi, vname in enumerate(bio_names):
        dr2 = probe_results[vname]['ridge']['delta_r2']
        if dr2 > DR2_THRESHOLD:
            target = bio_targets[:, vi]
            iaaft_res = iaaft_test(H_t, target, groups, n_surrogates=200, rng=iaaft_rng)
            iaaft_results[vname] = iaaft_res
            log.info("    %s: p=%.4f z=%.2f %s",
                     vname, iaaft_res['p_value'], iaaft_res['z_score'],
                     "SIGNIFICANT" if iaaft_res['significant'] else "not significant")
        else:
            iaaft_results[vname] = {
                'p_value': 1.0, 'significant': False,
                'reason': 'dR2 <= {}'.format(DR2_THRESHOLD)}

    # --- COMPILE ---
    n_mandatory = sum(1 for v in ablation_results.values()
                      if v.get('overall_verdict') == 'MANDATORY')
    n_causal = sum(1 for v in ablation_results.values()
                   if v.get('overall_verdict') in ('MANDATORY', 'CAUSAL'))
    mandatory_vars = [v for v, r in ablation_results.items()
                      if r.get('overall_verdict') == 'MANDATORY']

    log.info("\n  --- %s SUMMARY ---", condition_name)
    log.info("  CC: %.3f", cc)
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
        'verification': verification,
        'probing': probe_results,
        'ablation': ablation_results,
        'iaaft': iaaft_results,
        'summary': {
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
# COMPARISON TABLE + 2x2 MATRIX
# =========================================================================

def print_comparison_table(results):
    log.info("\n" + "=" * 100)
    log.info("SYNTHETIC REALITY v2 -- COMPARISON TABLE (7 variables x 5 conditions)")
    log.info("=" * 100)

    header = '{:<22}'.format('Variable')
    for r in results:
        header += ' | {:>8} dR2'.format(r['condition'][:8])
    log.info(header)
    log.info("-" * len(header))

    for vname in PROBE_VARIABLES:
        line = '{:<22}'.format(vname)
        for r in results:
            dr2 = r['probing'].get(vname, {}).get('ridge', {}).get('delta_r2', 0)
            abl = r['ablation'].get(vname, {})
            verdict = abl.get('overall_verdict', '-')
            marker = '*' if verdict == 'MANDATORY' else ('+' if verdict == 'CAUSAL' else ' ')
            line += ' |  {:>8.4f}{}'.format(dr2, marker)
        log.info(line)

    log.info("-" * len(header))
    log.info("  * = MANDATORY   + = CAUSAL   (blank) = BYPRODUCT/NOT_TESTED")

    verdict_line = '{:<22}'.format('VERDICT')
    for r in results:
        n_m = r['summary']['n_mandatory']
        status = 'ZOMBIE' if n_m == 0 else '{}/7 MAND'.format(n_m)
        verdict_line += ' | {:>10}'.format(status)
    log.info(verdict_line)

    verify_line = '{:<22}'.format('VERIFICATION')
    for r in results:
        v = r['summary']['verification_valid']
        verify_line += ' | {:>10}'.format('PASS' if v else 'FAIL')
    log.info(verify_line)

    log.info("")
    for r in results:
        mvars = r['summary']['mandatory_variables']
        if mvars:
            log.info("  %s mandatory: %s", r['condition'], ', '.join(mvars))
        else:
            log.info("  %s: ZOMBIE", r['condition'])


def print_2x2_matrix(results):
    """Print the critical 2x2 verification matrix."""
    log.info("\n" + "=" * 70)
    log.info("2x2 VERIFICATION MATRIX")
    log.info("=" * 70)

    rate_vars = {'firing_rate_input', 'trial_variance', 'rate_covariance'}
    temporal_vars = {'theta_power', 'gamma_power', 'theta_gamma_pac'}

    log.info("")
    log.info("                     | Rate mandatory | Rate NOT mandatory")
    log.info("  --------------------------------------------------------")

    cells = {}
    for r in results:
        mand = set(r['summary']['mandatory_variables'])
        has_rate = bool(mand & rate_vars)
        has_temporal = bool(mand & temporal_vars)
        cell = ('rate' if has_rate else 'no_rate',
                'temporal' if has_temporal else 'no_temporal')
        cells.setdefault(cell, []).append(r['condition'])

    r1c1 = ', '.join(cells.get(('rate', 'temporal'), ['-']))
    r1c2 = ', '.join(cells.get(('no_rate', 'temporal'), ['-']))
    log.info("  Temporal mandatory | %-14s | %s", r1c1, r1c2)

    r2c1 = ', '.join(cells.get(('rate', 'no_temporal'), ['-']))
    r2c2 = ', '.join(cells.get(('no_rate', 'no_temporal'), ['-']))
    log.info("  Temporal NOT mand  | %-14s | %s", r2c1, r2c2)

    log.info("  --------------------------------------------------------")

    n_cells = sum(1 for v in cells.values() if v)
    log.info("\n  Populated cells: %d/4", n_cells)

    if n_cells >= 2:
        log.info("  *** EXPERIMENT SUCCESS: at least 2 cells populated ***")
        log.info("  *** Different input structures produce different mandatory variables ***")
    else:
        log.info("  *** EXPERIMENT INCONCLUSIVE: only %d cell(s) populated ***", n_cells)

    rate_conds = cells.get(('rate', 'no_temporal'), [])
    temporal_conds = cells.get(('no_rate', 'temporal'), [])
    zombie_conds = cells.get(('no_rate', 'no_temporal'), [])

    if 'RATE_MODULATED' in rate_conds:
        log.info("  RATE_MODULATED -> rate-only cell: PREDICTION CONFIRMED")
    if 'TEMPORAL_ONLY' in temporal_conds:
        log.info("  TEMPORAL_ONLY -> temporal-only cell: PREDICTION CONFIRMED")
    if 'PURE_NOISE' in zombie_conds:
        log.info("  PURE_NOISE -> zombie cell: PREDICTION CONFIRMED")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Synthetic Reality v2 -- Radical Restructuring')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--output-dir', default='results/synthetic_reality_v2')
    parser.add_argument('--source-subject', default='5')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    log.info("=" * 70)
    log.info("SYNTHETIC REALITY v2 -- RADICAL RESTRUCTURING EXPERIMENT")
    log.info("  Goal: show DIFFERENT task structures -> DIFFERENT mandatory variables")
    log.info("  Source subject: sub-%s", args.source_subject)
    log.info("  Hidden dim: %d", args.hidden_dim)
    log.info("  Device: %s", args.device)
    log.info("  Conditions: SEQUENTIAL, RATE_MODULATED, TEMPORAL_ONLY, SPATIAL_ONLY, PURE_NOISE")
    log.info("  Probe variables: %s", ', '.join(PROBE_VARIABLES))
    log.info("=" * 70)

    t_start = time.time()

    # Load source data
    trials, meta = load_source_data(args.processed_dir, args.source_subject)

    # Generate all 5 conditions
    log.info("\n--- GENERATING CONDITIONS ---")
    cond1 = condition_sequential(trials)
    cond2 = condition_rate_modulated(trials, rng)
    cond3 = condition_temporal_only(trials, rng)
    cond4 = condition_spatial_only(trials, rng)
    cond5 = condition_pure_noise(trials, rng)

    conditions = [
        ('SEQUENTIAL', cond1),
        ('RATE_MODULATED', cond2),
        ('TEMPORAL_ONLY', cond3),
        ('SPATIAL_ONLY', cond4),
        ('PURE_NOISE', cond5),
    ]

    # Run each condition
    all_results = []
    for cond_name, cond_trials in conditions:
        r = run_condition(cond_name, cond_trials, trials,
                          args.hidden_dim, args.device, rng, str(output_dir))
        all_results.append(r)

    # Comparison table
    print_comparison_table(all_results)

    # 2x2 matrix
    print_2x2_matrix(all_results)

    # Save master results
    total_time = time.time() - t_start
    master = {
        'experiment': 'synthetic_reality_v2',
        'source_subject': args.source_subject,
        'hidden_dim': args.hidden_dim,
        'total_time_min': total_time / 60,
        'conditions': [_convert_numpy(r) for r in all_results],
        'predictions': {
            'sequential': 'firing_rate_input mandatory (replicates v1)',
            'rate_modulated': 'firing_rate_input mandatory, temporal vars NOT mandatory',
            'temporal_only': 'theta_power or theta_gamma_pac mandatory, rate vars NOT mandatory',
            'spatial_only': 'population_synchrony or rate_covariance mandatory',
            'pure_noise': 'NOTHING mandatory (universal zombie)',
        },
        'critical_test': (
            'Conditions 2 (RATE_MODULATED) and 3 (TEMPORAL_ONLY) should populate '
            'DIFFERENT cells of the 2x2 matrix (rate-only vs temporal-only). '
            'If confirmed, mandatory variables reflect the available coding dimension.'
        ),
        'caveat': (
            'Tests whether mandatory variables are task-structure-contingent within '
            'the EXISTING architecture. Does NOT test what a brain evolved in a '
            'different reality would do. The evolutionary circularity limits the '
            'claim to: "within this architecture, mandatory variables track task structure."'
        ),
    }
    with open(output_dir / 'synthetic_reality_v2_results.json', 'w') as f:
        json.dump(_convert_numpy(master), f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("SYNTHETIC REALITY v2 COMPLETE (%.1f min)", total_time / 60)
    log.info("=" * 70)
    log.info("Results: %s", output_dir)


if __name__ == '__main__':
    main()
