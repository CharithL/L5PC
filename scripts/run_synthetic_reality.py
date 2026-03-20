#!/usr/bin/env python3
"""
run_synthetic_reality.py

DESCARTES Synthetic Reality Experiment — Task Structure Drives Mandatory Variables

Philosophical claim: mandatory variables are contingent on reality structure,
not cosmically fixed. theta_gamma_pac is mandatory because the Sternberg task
presents sequential items requiring temporal multiplexing. If we change the
input structure while holding the surrogate architecture constant, mandatory
variables should shift.

Conditions:
  1. SEQUENTIAL (control) — real Sternberg data, replicates Circuit 6
  2. BOUNDARYLESS — preserves marginal stats, destroys trial structure
  3. PURE_NOISE — Poisson noise matched to real rate/variance, zero structure

Predictions:
  1. theta_gamma_pac mandatory (replicates Circuit 6)
  2. theta_gamma_pac NOT mandatory; possibly rate covariance emerges instead
  3. NOTHING mandatory — universal zombie

For each condition: train LSTM (h=64), Ridge delta-R2 probing (GroupKFold),
resample ablation, iAAFT statistical hardening.

Usage:
  python scripts/run_synthetic_reality.py \
    --processed-dir data/kyzar_processed \
    --output-dir results/synthetic_reality \
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('synthetic_reality')

PROBE_VARIABLES = [
    'firing_rate_input', 'trial_variance', 'theta_power',
    'gamma_power', 'theta_gamma_pac',
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
# MODEL (same as Phase 2)
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
                'X': X_data[key],
                'Y': Y_data[key],
                'epoch': E_data[key],
                'meta': meta['trial_metadata'][ti],
            })

    log.info("Loaded %d trials from sub-%s", len(trials), subject)
    log.info("  n_in=%d, n_out=%d", trials[0]['X'].shape[1], trials[0]['Y'].shape[1])
    log.info("  trial lengths: %d-%d timesteps",
             min(t['X'].shape[0] for t in trials),
             max(t['X'].shape[0] for t in trials))

    return trials, meta


# =========================================================================
# CONDITION GENERATORS
# =========================================================================

def condition_sequential(trials):
    """Condition 1: Use real data as-is (control)."""
    log.info("  Condition 1: SEQUENTIAL (real Sternberg data, %d trials)", len(trials))
    return trials


def condition_boundaryless(trials, rng):
    """Condition 2: Preserve marginal stats, destroy trial structure.

    Strategy:
    1. Concatenate all trials into one long stream
    2. Cut into random-length segments (preserves local autocorrelation)
    3. Shuffle segment order (destroys trial boundaries)
    4. Apply random temporal stretching/compression
    5. Re-cut into pseudo-trials of random length
    """
    log.info("  Condition 2: BOUNDARYLESS (destroy trial structure)")

    all_X = np.concatenate([t['X'] for t in trials], axis=0)
    all_Y = np.concatenate([t['Y'] for t in trials], axis=0)
    all_epoch = np.concatenate([t['epoch'] for t in trials], axis=0)
    T_total = len(all_X)

    # Cut into random-sized segments (50-200 timesteps)
    segment_lengths = []
    pos = 0
    while pos < T_total:
        seg_len = rng.integers(50, 200)
        seg_len = min(seg_len, T_total - pos)
        if seg_len < 10:
            if segment_lengths:
                segment_lengths[-1] += seg_len
            pos += seg_len
            break
        segment_lengths.append(seg_len)
        pos += seg_len

    segments_X, segments_Y, segments_E = [], [], []
    pos = 0
    for sl in segment_lengths:
        segments_X.append(all_X[pos:pos+sl])
        segments_Y.append(all_Y[pos:pos+sl])
        segments_E.append(all_epoch[pos:pos+sl])
        pos += sl

    # Shuffle segment order
    n_segs = len(segments_X)
    shuffle_idx = rng.permutation(n_segs)
    segments_X = [segments_X[i] for i in shuffle_idx]
    segments_Y = [segments_Y[i] for i in shuffle_idx]
    segments_E = [segments_E[i] for i in shuffle_idx]

    # Temporal stretching/compression per segment
    stretched_X, stretched_Y, stretched_E = [], [], []
    for sx, sy, se in zip(segments_X, segments_Y, segments_E):
        factor = rng.uniform(0.7, 1.4)
        new_len = max(10, int(len(sx) * factor))
        old_t = np.linspace(0, 1, len(sx))
        new_t = np.linspace(0, 1, new_len)

        new_x = np.zeros((new_len, sx.shape[1]))
        for d in range(sx.shape[1]):
            new_x[:, d] = np.interp(new_t, old_t, sx[:, d])

        new_y = np.zeros((new_len, sy.shape[1]))
        for d in range(sy.shape[1]):
            new_y[:, d] = np.interp(new_t, old_t, sy[:, d])

        new_e = np.interp(new_t, old_t, se.astype(float))
        new_e = np.round(new_e).astype(int)

        stretched_X.append(new_x)
        stretched_Y.append(new_y)
        stretched_E.append(new_e)

    # Re-concatenate and re-cut into pseudo-trials
    all_X_new = np.concatenate(stretched_X, axis=0)
    all_Y_new = np.concatenate(stretched_Y, axis=0)
    all_E_new = np.concatenate(stretched_E, axis=0)
    T_new = len(all_X_new)

    n_trials_out = len(trials)
    max_splits = min(n_trials_out - 1, T_new // 50 - 1)
    if max_splits < 1:
        max_splits = 1
    split_points = sorted(rng.choice(range(50, T_new - 50),
                                     size=max_splits,
                                     replace=False))
    split_points = [0] + list(split_points) + [T_new]

    new_trials = []
    for i in range(len(split_points) - 1):
        s, e = split_points[i], split_points[i+1]
        if e - s < 20:
            continue
        new_trials.append({
            'X': all_X_new[s:e].astype(np.float32),
            'Y': all_Y_new[s:e].astype(np.float32),
            'epoch': all_E_new[s:e],
            'meta': {'load': 0, 'accuracy': 0, 'reaction_time_ms': 0,
                     'probe_in_out': 0, 'enc1_pic': 0},
        })

    log.info("    %d segments shuffled+stretched -> %d pseudo-trials (%d timesteps)",
             n_segs, len(new_trials), T_new)

    orig_mean = np.mean(all_X)
    new_mean = np.mean(all_X_new)
    orig_std = np.std(all_X)
    new_std = np.std(all_X_new)
    log.info("    Marginal stats: mean %.4f->%.4f, std %.4f->%.4f",
             orig_mean, new_mean, orig_std, new_std)

    return new_trials


def condition_pure_noise(trials, rng):
    """Condition 3: Poisson noise matched to real statistics.

    No temporal structure, no trial structure, no oscillatory modulation.
    Independent across neurons, matched mean and variance.
    """
    log.info("  Condition 3: PURE_NOISE (Poisson matched to statistics)")

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
            'meta': {'load': 0, 'accuracy': 0, 'reaction_time_ms': 0,
                     'probe_in_out': 0, 'enc1_pic': 0},
        })

    log.info("    Generated %d noise trials matched to real stats", len(noise_trials))
    log.info("    X: mean=%.4f (real=%.4f), Y: mean=%.4f (real=%.4f)",
             np.mean(np.concatenate([t['X'] for t in noise_trials])),
             np.mean(all_X),
             np.mean(np.concatenate([t['Y'] for t in noise_trials])),
             np.mean(all_Y))

    return noise_trials


# =========================================================================
# BIO VARIABLE COMPUTATION (5 key variables)
# =========================================================================

def compute_bio_variables(X_trial, Y_trial, epoch_mask, trial_meta, dt_ms=10):
    """Compute the 5 key biological variables for one trial."""
    T = X_trial.shape[0]
    dt_s = dt_ms / 1000.0
    fs = 1.0 / dt_s

    variables = {}

    variables['firing_rate_input'] = np.mean(X_trial, axis=1)

    win = max(1, int(200 / dt_ms))
    fr_all = np.concatenate([X_trial, Y_trial], axis=1)
    mean_fr = np.mean(fr_all, axis=1)
    variables['trial_variance'] = _rolling_var(mean_fr, win)

    pop_rate = np.mean(X_trial, axis=1)
    variables['theta_power'] = _bandpower(pop_rate, fs, 4, 8)
    variables['gamma_power'] = _bandpower(pop_rate, fs, 30, min(80, fs/2 - 1))
    variables['theta_gamma_pac'] = _pac_strength(pop_rate, fs)

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
    """Convert trial list to X_data/Y_data dicts."""
    X_data = {}
    Y_data = {}
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
# PROBING + ABLATION
# =========================================================================

def ridge_delta_r2(H_trained, H_untrained, target, groups, alpha=1.0):
    gkf = GroupKFold(n_splits=5)
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


# =========================================================================
# iAAFT HARDENING
# =========================================================================

def iaaft_surrogate(x, rng, n_iter=100):
    """Iteratively Adjusted Amplitude and Fourier Transform surrogate.
    Preserves both amplitude distribution and power spectrum."""
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
    """Test whether R2 is significant vs iAAFT surrogates of the target."""
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

def run_condition(condition_name, trials, hidden_dim, device, rng, output_dir):
    """Full pipeline for one condition: train -> probe -> ablate -> harden."""
    log.info("\n" + "=" * 70)
    log.info("CONDITION: %s", condition_name)
    log.info("=" * 70)

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
    log.info("  Computing biological variables...")
    bio_all, trial_groups_all = [], []
    h_tr_all, h_un_all = [], []
    bio_names = None

    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key not in h_trained:
            continue

        bio, names = compute_bio_variables(
            trials[ti]['X'], trials[ti]['Y'],
            trials[ti]['epoch'], trials[ti]['meta'])

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
    bio_targets = np.concatenate(bio_all, axis=0).astype(np.float64)
    groups = np.array(trial_groups_all)

    log.info("  Samples: %d, Groups: %d unique trials", len(H_t), len(np.unique(groups)))

    # --- PROBE ---
    log.info("  Probing %d variables...", len(bio_names))
    probe_results = {}
    for vi, vname in enumerate(bio_names):
        target = bio_targets[:, vi]
        ridge = ridge_delta_r2(H_t, H_u, target, groups)
        log.info("    %s: dR2=%.4f (trained=%.4f, untrained=%.4f)",
                 vname, ridge['delta_r2'], ridge['r2_trained'], ridge['r2_untrained'])
        probe_results[vname] = {'ridge': ridge}

    # --- ABLATION (on variables with dR2 > 0.05) ---
    log.info("  Resample ablation...")
    ablation_results = {}
    abl_rng = np.random.default_rng(42)
    for vi, vname in enumerate(bio_names):
        dr2 = probe_results[vname]['ridge']['delta_r2']
        if dr2 > 0.05:
            target = bio_targets[:, vi]
            abl = resample_ablation(H_t, target, groups, rng=abl_rng)
            ablation_results[vname] = abl
            log.info("    %s: %s (z-scores: %s)",
                     vname, abl['overall_verdict'],
                     ', '.join('{:.1f}'.format(r['z_score'])
                               for r in abl['per_k'].values()))
        else:
            ablation_results[vname] = {
                'overall_verdict': 'NOT_TESTED', 'reason': 'dR2 <= 0.05'}

    # --- iAAFT HARDENING (on variables with dR2 > 0.05) ---
    log.info("  iAAFT statistical hardening...")
    iaaft_results = {}
    iaaft_rng = np.random.default_rng(123)
    for vi, vname in enumerate(bio_names):
        dr2 = probe_results[vname]['ridge']['delta_r2']
        if dr2 > 0.05:
            target = bio_targets[:, vi]
            iaaft_res = iaaft_test(H_t, target, groups, n_surrogates=200, rng=iaaft_rng)
            iaaft_results[vname] = iaaft_res
            log.info("    %s: p=%.4f z=%.2f %s",
                     vname, iaaft_res['p_value'], iaaft_res['z_score'],
                     "SIGNIFICANT" if iaaft_res['significant'] else "not significant")
        else:
            iaaft_results[vname] = {
                'p_value': 1.0, 'significant': False,
                'reason': 'dR2 <= 0.05'}

    # --- COMPILE RESULTS ---
    n_mandatory = sum(1 for v in ablation_results.values()
                      if v.get('overall_verdict') == 'MANDATORY')
    n_causal = sum(1 for v in ablation_results.values()
                   if v.get('overall_verdict') in ('MANDATORY', 'CAUSAL'))
    mandatory_vars = [v for v, r in ablation_results.items()
                      if r.get('overall_verdict') == 'MANDATORY']

    log.info("\n  --- %s SUMMARY ---", condition_name)
    log.info("  CC: %.3f", cc)
    log.info("  Mandatory: %d/5 -- %s", n_mandatory,
             ', '.join(mandatory_vars) if mandatory_vars else 'NONE (ZOMBIE)')
    log.info("  Causal: %d/5", n_causal)

    result = {
        'condition': condition_name,
        'output_cc': cc,
        'n_epochs': n_epochs,
        'n_trials': n_trials,
        'n_samples': len(H_t),
        'hidden_dim': hidden_dim,
        'probing': probe_results,
        'ablation': ablation_results,
        'iaaft': iaaft_results,
        'summary': {
            'n_mandatory': n_mandatory,
            'n_causal': n_causal,
            'mandatory_variables': mandatory_vars,
        },
    }

    # Save per-condition
    cond_dir = Path(output_dir) / condition_name.lower()
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / 'results.json', 'w') as f:
        json.dump(_convert_numpy(result), f, indent=2)

    return result


# =========================================================================
# COMPARISON TABLE + MAIN
# =========================================================================

def print_comparison_table(results):
    """Print the final Condition x Variable comparison table."""
    log.info("\n" + "=" * 90)
    log.info("SYNTHETIC REALITY EXPERIMENT -- COMPARISON TABLE")
    log.info("=" * 90)

    header = '{:<25}'.format('Variable')
    for r in results:
        header += ' | {:>12} dR2'.format(r['condition'][:12])
        header += ' | {:>8}'.format('iAAFT p')
        header += ' | {:>6}'.format('z-abl')
    log.info(header)
    log.info("-" * len(header))

    for vname in PROBE_VARIABLES:
        line = '{:<25}'.format(vname)
        for r in results:
            dr2 = r['probing'].get(vname, {}).get('ridge', {}).get('delta_r2', 0)
            p = r['iaaft'].get(vname, {}).get('p_value', 1.0)

            abl = r['ablation'].get(vname, {})
            per_k = abl.get('per_k', {})
            z_scores = [v.get('z_score', 0) for v in per_k.values()] if per_k else [0]
            best_z = min(z_scores)

            line += ' | {:>15.4f}'.format(dr2)
            line += ' | {:>8.4f}'.format(p)
            line += ' | {:>6.1f}'.format(best_z)
        log.info(line)

    log.info("-" * len(header))

    verdict_line = '{:<25}'.format('VERDICT')
    for r in results:
        n_m = r['summary']['n_mandatory']
        if n_m == 0:
            verdict_line += ' | {:>15}   |          |       '.format('ZOMBIE')
        else:
            verdict_line += ' | {:>15}   |          |       '.format(
                str(n_m) + '/5 MAND')
    log.info(verdict_line)

    for r in results:
        mvars = r['summary']['mandatory_variables']
        if mvars:
            log.info("  %s mandatory: %s", r['condition'], ', '.join(mvars))
        else:
            log.info("  %s: UNIVERSAL ZOMBIE -- no mandatory variables", r['condition'])


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Synthetic Reality Experiment')
    parser.add_argument('--processed-dir', required=True,
                        help='Path to kyzar_processed directory')
    parser.add_argument('--output-dir', default='results/synthetic_reality')
    parser.add_argument('--source-subject', default='5',
                        help='Subject to use as source (default: 5, strongest non-zombie)')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    log.info("=" * 70)
    log.info("SYNTHETIC REALITY EXPERIMENT")
    log.info("  Claim: mandatory variables are contingent on task structure")
    log.info("  Source subject: sub-%s", args.source_subject)
    log.info("  Hidden dim: %d", args.hidden_dim)
    log.info("  Device: %s", args.device)
    log.info("=" * 70)

    t_start = time.time()

    # Load source data
    trials, meta = load_source_data(args.processed_dir, args.source_subject)

    # Generate conditions
    cond1_trials = condition_sequential(trials)
    cond2_trials = condition_boundaryless(trials, rng)
    cond3_trials = condition_pure_noise(trials, rng)

    # Run each condition
    all_results = []

    r1 = run_condition('SEQUENTIAL', cond1_trials, args.hidden_dim,
                       args.device, rng, str(output_dir))
    all_results.append(r1)

    r2 = run_condition('BOUNDARYLESS', cond2_trials, args.hidden_dim,
                       args.device, rng, str(output_dir))
    all_results.append(r2)

    r3 = run_condition('PURE_NOISE', cond3_trials, args.hidden_dim,
                       args.device, rng, str(output_dir))
    all_results.append(r3)

    # Comparison table
    print_comparison_table(all_results)

    # Save master results
    total_time = time.time() - t_start
    master = {
        'experiment': 'synthetic_reality',
        'source_subject': args.source_subject,
        'hidden_dim': args.hidden_dim,
        'total_time_min': total_time / 60,
        'conditions': [_convert_numpy(r) for r in all_results],
        'predictions': {
            'sequential': 'theta_gamma_pac mandatory (replicates Circuit 6)',
            'boundaryless': 'theta_gamma_pac NOT mandatory; rate covariance may emerge',
            'pure_noise': 'NOTHING mandatory -- universal zombie',
        },
        'caveat': (
            'This tests whether mandatory variables are task-structure-contingent '
            'within the EXISTING architecture. It does NOT test what a brain evolved '
            'in a different reality would do -- that brain would have different '
            'architecture entirely. The evolutionary circularity limits the claim '
            'to: "within this architecture, mandatory variables track task structure."'
        ),
    }
    with open(output_dir / 'synthetic_reality_results.json', 'w') as f:
        json.dump(_convert_numpy(master), f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE (%.1f min)", total_time / 60)
    log.info("=" * 70)
    log.info("Results: %s", output_dir)

    # Final interpretation
    c1_mand = r1['summary']['n_mandatory']
    c2_mand = r2['summary']['n_mandatory']
    c3_mand = r3['summary']['n_mandatory']

    if c1_mand > 0 and c2_mand < c1_mand and c3_mand == 0:
        log.info("\n*** FULL PREDICTION CONFIRMED: Sequential > Boundaryless > Noise ***")
        log.info("*** Task structure drives mandatory variable emergence ***")
    elif c1_mand > 0 and c3_mand == 0:
        log.info("\n*** PREDICTION CONFIRMED: Structured reality (%d mandatory) vs "
                 "noise (0 mandatory) ***", c1_mand)
        log.info("*** Mandatory variables REQUIRE structured reality ***")
    elif c1_mand == 0:
        log.info("\n*** UNEXPECTED: Control condition also zombie -- "
                 "source subject may not replicate ***")
    else:
        log.info("\n*** MIXED RESULT: Seq=%d, Boundaryless=%d, Noise=%d mandatory ***",
                 c1_mand, c2_mand, c3_mand)
        log.info("*** Requires further investigation ***")


if __name__ == '__main__':
    main()
