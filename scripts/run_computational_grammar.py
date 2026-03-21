#!/usr/bin/env python3
"""
run_computational_grammar.py

DESCARTES Computational Grammar Analysis — Shared Abstract Properties

Compares biological mandatory variables (from non-zombie subjects sub-2, sub-5, sub-14)
with alien mandatory variables (from zombie subjects sub-8, sub-10, sub-11 discovered
by run_zombie_structure.py) across 5 abstract computational properties:

  1. Linearity:           Can the component be reconstructed from linear input projection?
  2. Temporal extent:     Autocorrelation decay timescale (instantaneous to slow drift)
  3. Population emergence: Does it depend on the full population or single neurons?
  4. Cross-dimensional:   Do mandatory dims interact nonlinearly (superadditivity)?
  5. Epoch sensitivity:   Is it causal across all task epochs or epoch-specific?

If alien and biological mandatory variables share these properties then a computational
grammar of consciousness has been discovered. If they differ then mandatory variables are
idiosyncratic to each computational solution.

Prerequisites:
  - Completed zombie structure analysis (run_zombie_structure.py)
  - Phase 2 models for biological subjects (sub-2, sub-5, sub-14)
  - Phase 3-4 probing results identifying biological mandatory variables

Usage:
  python scripts/run_computational_grammar.py \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --zombie-results-dir results/kyzar/zombie_structure \
    --phase34-results-dir results/kyzar \
    --output-dir results/kyzar/computational_grammar \
    --zombie-subjects 8 10 11 \
    --bio-subjects 2 5 14 \
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('computational_grammar')


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
# MODEL (match Phase 2)
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
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                         enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        pred = self.output_layer(out)
        return pred, out


# =========================================================================
# DATA LOADING UTILITIES
# =========================================================================

def load_session_data(processed_dir, model_dir, session_name, device='cpu'):
    """Load hidden states, input/output data, and model for a session."""
    session_proc = Path(processed_dir) / session_name
    session_model = Path(model_dir) / session_name

    with open(session_proc / 'metadata.json') as f:
        meta = json.load(f)
    n_trials = meta['n_trials']

    # Find best model
    best_hdim, best_cc = None, -1
    for hdir in sorted(session_model.glob('lstm_h*')):
        val_path = hdir / 'output_validation.json'
        if not val_path.exists():
            continue
        with open(val_path) as f:
            val_data = json.load(f)
        if val_data.get('passed_gate', False) and val_data['output_cc'] > best_cc:
            best_cc = val_data['output_cc']
            best_hdim = val_data['hidden_dim']

    if best_hdim is None:
        hdirs = sorted(session_model.glob('lstm_h*'))
        if not hdirs:
            return None
        best_hdim = int(hdirs[-1].name.replace('lstm_h', ''))

    model_path = session_model / 'lstm_h{}'.format(best_hdim)

    # Load hidden states
    h_path = model_path / 'hidden_trained.npz'
    if not h_path.exists():
        return None
    h_dict = dict(np.load(h_path))

    # Flatten
    h_arrays, groups_list = [], []
    X_arrays, Y_arrays, epoch_arrays = [], [], []

    X_data = dict(np.load(session_proc / 'X_trials.npz'))
    Y_data = dict(np.load(session_proc / 'Y_trials.npz'))
    E_path = session_proc / 'epoch_masks.npz'
    E_data = dict(np.load(E_path)) if E_path.exists() else {}

    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key in h_dict and key in X_data:
            h = h_dict[key]
            T_h = h.shape[0]
            h_arrays.append(h)
            groups_list.extend([ti] * T_h)
            X_arrays.append(X_data[key][:T_h])
            Y_arrays.append(Y_data.get(key, np.zeros((T_h, 1)))[:T_h])
            epoch_arrays.append(E_data[key][:T_h] if key in E_data else np.zeros(T_h, dtype=int))

    if not h_arrays:
        return None

    H = np.concatenate(h_arrays, axis=0).astype(np.float64)
    groups = np.array(groups_list)
    X_flat = np.concatenate(X_arrays, axis=0)
    Y_flat = np.concatenate(Y_arrays, axis=0)
    epoch_flat = np.concatenate(epoch_arrays, axis=0)

    # Load model
    checkpoint = model_path / 'trained_model.pt'
    model = None
    if checkpoint.exists():
        n_in, n_out = X_flat.shape[1], Y_flat.shape[1]
        model = LSTMSurrogate(n_in, n_out, best_hdim).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval_flag = True

    return {
        'session': session_name,
        'H': H, 'groups': groups,
        'X': X_flat, 'Y': Y_flat, 'epochs': epoch_flat,
        'model': model, 'hidden_dim': best_hdim,
        'n_trials': len(np.unique(groups)),
    }


def get_biological_mandatory_target(data):
    """Compute theta_gamma_pac from raw input for biological subjects."""
    from scipy.signal import butter, filtfilt, hilbert

    X = data['X']
    pop_rate = np.mean(X, axis=1)
    fs = 100.0  # 10ms bins = 100 Hz
    T = len(pop_rate)

    try:
        nyq = fs / 2
        b, a = butter(3, [4/nyq, 8/nyq], btype='band')
        padlen = min(3 * max(len(b), len(a)), T - 1)
        theta_filt = filtfilt(b, a, pop_rate, padlen=padlen)
        theta_phase = np.angle(hilbert(theta_filt))

        gmax = min(80, fs/2 - 1)
        b2, a2 = butter(3, [30/nyq, gmax/nyq], btype='band')
        gamma_filt = filtfilt(b2, a2, pop_rate, padlen=padlen)
        gamma_amp = np.abs(hilbert(gamma_filt))

        win = 50
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


def get_alien_mandatory_components(zombie_results_path, data):
    """Load mandatory components from zombie structure analysis."""
    if not Path(zombie_results_path).exists():
        return [], {}

    with open(zombie_results_path) as f:
        zr = json.load(f)

    mandatory_names = zr.get('mandatory_components', [])
    if not mandatory_names:
        return [], {}

    H = data['H']

    # Reconstruct components
    components = {}
    for name in mandatory_names:
        if name.startswith('PCA_'):
            idx = int(name.split('_')[1])
            n_comp = max(idx + 1, 10)
            n_comp = min(n_comp, H.shape[1], H.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            proj = pca.fit_transform(H)
            if idx < proj.shape[1]:
                components[name] = proj[:, idx]
        elif name.startswith('ICA_'):
            from sklearn.decomposition import FastICA
            idx = int(name.split('_')[1])
            n_comp = max(idx + 1, 10)
            n_comp = min(n_comp, H.shape[1])
            try:
                ica = FastICA(n_components=n_comp, max_iter=500, random_state=42)
                proj = ica.fit_transform(H)
                if idx < proj.shape[1]:
                    components[name] = proj[:, idx]
            except Exception:
                pass

    return mandatory_names, components


# =========================================================================
# PROPERTY 1: LINEARITY TEST
# =========================================================================

def linearity_test(component_values, input_data, groups, n_splits=5):
    """Can this mandatory component be reconstructed from LINEAR input projection?

    High R2 (> 0.8) = linear, simple random projection could do it
    Low R2 (< 0.5) = nonlinear, requires learned computation
    """
    if np.std(component_values) < 1e-10:
        return 0.0

    component_values = np.nan_to_num(component_values, nan=0.0)
    n_groups = len(np.unique(groups))
    n_sp = min(n_splits, n_groups)
    if n_sp < 2:
        return 0.0

    gkf = GroupKFold(n_splits=n_sp)

    try:
        scores = cross_val_score(
            Ridge(alpha=1.0), input_data, component_values,
            cv=gkf, groups=groups, scoring='r2')
        return float(np.mean(scores))
    except Exception:
        return 0.0


# =========================================================================
# PROPERTY 2: TEMPORAL EXTENT
# =========================================================================

def temporal_extent(component_timeseries, dt_ms=10.0):
    """Autocorrelation decay timescale.

    Short (< 5ms):    instantaneous, no memory needed
    Medium (5-50ms):  gamma timescale
    Long (50-250ms):  theta timescale
    Very long (>250ms): slow drift / epoch-level
    """
    n = len(component_timeseries)
    max_lag = min(int(500 / dt_ms), n // 4)
    if max_lag < 2:
        return 0.0, 'instantaneous'

    x = component_timeseries - np.mean(component_timeseries)
    var = np.var(x)
    if var < 1e-10:
        return 0.0, 'instantaneous'

    ac = np.zeros(max_lag + 1)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = np.mean(x[:-lag] * x[lag:]) / var

    threshold = 1.0 / np.e
    decay_lag = max_lag
    for lag in range(1, max_lag + 1):
        if ac[lag] < threshold:
            decay_lag = lag
            break

    decay_ms = float(decay_lag * dt_ms)

    if decay_ms < 5:
        category = 'instantaneous'
    elif decay_ms < 50:
        category = 'gamma_timescale'
    elif decay_ms < 250:
        category = 'theta_timescale'
    else:
        category = 'slow_drift'

    return decay_ms, category


# =========================================================================
# PROPERTY 3: POPULATION EMERGENCE
# =========================================================================

def population_emergence(component_values, input_data, n_sample=10000):
    """Does this component depend on the full population or single neurons?

    ratio > 2:  population-emergent (irreducible to single neurons)
    ratio ~ 1:  single-unit driven
    """
    n = min(n_sample, len(component_values))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(component_values), n, replace=False)
    comp_sub = component_values[idx]
    input_sub = input_data[idx]

    if np.std(comp_sub) < 1e-10:
        return 1.0, 0.0, []

    n_neurons = input_sub.shape[1]
    n_check = min(n_neurons, 50)

    single_mis = []
    for i in range(n_check):
        try:
            mi = mutual_info_regression(
                input_sub[:, i:i+1], comp_sub,
                n_neighbors=5, random_state=42)[0]
            single_mis.append(float(mi))
        except Exception:
            single_mis.append(0.0)

    top_neurons = min(30, n_neurons)
    try:
        pop_mi = mutual_info_regression(
            input_sub[:, :top_neurons], comp_sub,
            n_neighbors=5, random_state=42)[0]
        pop_mi = float(pop_mi)
    except Exception:
        pop_mi = 0.0

    max_single = max(single_mis) if single_mis else 0.0
    ratio = pop_mi / (max_single + 1e-10)

    return ratio, pop_mi, single_mis


# =========================================================================
# PROPERTY 4: CROSS-DIMENSIONAL COUPLING
# =========================================================================

def cross_dimensional_coupling(H, mandatory_directions, output_layer,
                                device='cpu', n_resamples=10):
    """Does ablating all mandatory dims together degrade more than sum of individual?

    ratio > 1.5: cross-dimensional coupling (nonlinear interaction)
    ratio ~ 1:   independent contributions
    """
    if len(mandatory_directions) < 2:
        return None

    hidden_dim = H.shape[1]
    rng = np.random.default_rng(42)

    with torch.no_grad():
        H_t = torch.tensor(H.astype(np.float32), device=device)
        baseline = output_layer(H_t).cpu().numpy()

    baseline_var = float(np.var(baseline))
    if baseline_var < 1e-10:
        return None

    individual_drops = []
    for dir_vec in mandatory_directions:
        dir_vec = np.array(dir_vec, dtype=np.float64)
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-10)

        drops = []
        for _ in range(n_resamples):
            H_abl = H.copy()
            proj = H_abl @ dir_vec
            H_abl -= np.outer(proj, dir_vec)
            shuffle_idx = rng.permutation(len(H))
            donor_proj = H[shuffle_idx] @ dir_vec
            H_abl += np.outer(donor_proj, dir_vec)

            with torch.no_grad():
                abl_out = output_layer(
                    torch.tensor(H_abl.astype(np.float32), device=device)
                ).cpu().numpy()
            mse = float(np.mean((abl_out - baseline) ** 2))
            drops.append(mse / baseline_var)

        individual_drops.append(float(np.mean(drops)))

    joint_drops = []
    for _ in range(n_resamples):
        H_abl = H.copy()
        for dir_vec in mandatory_directions:
            dir_vec = np.array(dir_vec, dtype=np.float64)
            dir_vec /= (np.linalg.norm(dir_vec) + 1e-10)
            proj = H_abl @ dir_vec
            H_abl -= np.outer(proj, dir_vec)
            shuffle_idx = rng.permutation(len(H))
            donor_proj = H[shuffle_idx] @ dir_vec
            H_abl += np.outer(donor_proj, dir_vec)

        with torch.no_grad():
            abl_out = output_layer(
                torch.tensor(H_abl.astype(np.float32), device=device)
            ).cpu().numpy()
        mse = float(np.mean((abl_out - baseline) ** 2))
        joint_drops.append(mse / baseline_var)

    joint_drop = float(np.mean(joint_drops))
    sum_individual = sum(individual_drops)
    ratio = joint_drop / (sum_individual + 1e-10)

    return {
        'individual_drops': individual_drops,
        'joint_drop': joint_drop,
        'sum_individual': sum_individual,
        'superadditivity_ratio': ratio,
        'cross_coupled': ratio > 1.5,
    }


# =========================================================================
# PROPERTY 5: EPOCH SENSITIVITY
# =========================================================================

def epoch_sensitivity(component_values, H, groups, epochs, output_layer,
                      hidden_dim, device='cpu', n_resamples=10):
    """Is the component causal across all task epochs or epoch-specific?"""
    epoch_names = {0: 'fixation', 1: 'encoding', 2: 'maintenance',
                   3: 'probe', 4: 'response'}

    unique_epochs = np.unique(epochs)
    results = {}
    rng = np.random.default_rng(42)

    corrs = np.array([np.corrcoef(H[:, d], component_values)[0, 1]
                      if np.std(H[:, d]) > 1e-10 else 0.0
                      for d in range(hidden_dim)])
    corrs = np.nan_to_num(corrs, nan=0.0)
    direction = corrs / (np.linalg.norm(corrs) + 1e-10)

    with torch.no_grad():
        H_t = torch.tensor(H.astype(np.float32), device=device)
        baseline = output_layer(H_t).cpu().numpy()

    for ep in unique_epochs:
        ep_mask = epochs == ep
        ep_name = epoch_names.get(int(ep), 'epoch_{}'.format(int(ep)))
        n_ep = int(np.sum(ep_mask))

        if n_ep < 50:
            results[ep_name] = {'mandatory': False, 'z_score': 0.0, 'n_samples': n_ep}
            continue

        targeted = []
        for _ in range(n_resamples):
            H_abl = H.copy()
            proj = H_abl[ep_mask] @ direction
            donor_idx = rng.choice(len(H), n_ep, replace=True)
            donor_proj = H[donor_idx] @ direction
            H_abl[ep_mask] = (H_abl[ep_mask]
                               - np.outer(proj, direction)
                               + np.outer(donor_proj[:n_ep], direction))

            with torch.no_grad():
                abl_out = output_layer(
                    torch.tensor(H_abl.astype(np.float32), device=device)
                ).cpu().numpy()
            mse = float(np.mean((abl_out[ep_mask] - baseline[ep_mask]) ** 2))
            targeted.append(mse)

        random_vals = []
        for _ in range(n_resamples):
            rand_dir = rng.standard_normal(hidden_dim)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)
            H_rand = H.copy()
            proj = H_rand[ep_mask] @ rand_dir
            H_rand[ep_mask] -= np.outer(proj, rand_dir)

            with torch.no_grad():
                rand_out = output_layer(
                    torch.tensor(H_rand.astype(np.float32), device=device)
                ).cpu().numpy()
            mse = float(np.mean((rand_out[ep_mask] - baseline[ep_mask]) ** 2))
            random_vals.append(mse)

        targ_mean = float(np.mean(targeted))
        rand_mean = float(np.mean(random_vals))
        rand_std = float(np.std(random_vals)) + 1e-10
        z = (targ_mean - rand_mean) / rand_std

        results[ep_name] = {
            'mandatory': z > 2.0,
            'z_score': float(z),
            'n_samples': n_ep,
        }

    n_mandatory_epochs = sum(1 for r in results.values() if r.get('mandatory', False))
    n_total_epochs = len(results)

    return {
        'per_epoch': results,
        'n_mandatory_epochs': n_mandatory_epochs,
        'n_total_epochs': n_total_epochs,
        'epoch_universal': n_mandatory_epochs >= min(4, n_total_epochs),
    }


# =========================================================================
# ANALYZE ONE SUBJECT
# =========================================================================

def analyze_subject(data, component_name, component_values,
                    is_biological, device='cpu'):
    """Run all 5 property tests on one mandatory component."""
    H = data['H']
    X = data['X']
    groups = data['groups']
    epochs = data['epochs']
    hidden_dim = data['hidden_dim']
    model = data['model']
    output_layer = model.output_layer if model else None

    log.info("  Analyzing %s (%s)...",
             component_name, 'biological' if is_biological else 'alien')

    # 1. Linearity
    lin_r2 = linearity_test(component_values, X, groups)
    nonlinear = lin_r2 < 0.5
    log.info("    Linearity R2: %.3f (%s)",
             lin_r2, "NONLINEAR" if nonlinear else "linear")

    # 2. Temporal extent
    decay_ms, category = temporal_extent(component_values, dt_ms=10.0)
    temporally_extended = decay_ms > 25.0
    log.info("    Temporal extent: %.1fms (%s) %s",
             decay_ms, category,
             "EXTENDED" if temporally_extended else "short")

    # 3. Population emergence
    ratio, pop_mi, single_mis = population_emergence(component_values, X)
    population_emergent = ratio > 2.0
    log.info("    Population emergence: ratio=%.2f (pop_MI=%.3f, max_single=%.3f) %s",
             ratio, pop_mi, max(single_mis) if single_mis else 0.0,
             "EMERGENT" if population_emergent else "single-unit")

    # 4. Cross-dimensional coupling (N/A for single component per subject)
    cross_coupled = None
    log.info("    Cross-dimensional: N/A (single component per subject)")

    # 5. Epoch sensitivity
    epoch_result = None
    epoch_universal = None
    if output_layer is not None:
        epoch_result = epoch_sensitivity(
            component_values, H, groups, epochs, output_layer,
            hidden_dim, device=device)
        epoch_universal = epoch_result['epoch_universal']
        log.info("    Epoch sensitivity: %d/%d mandatory %s",
                 epoch_result['n_mandatory_epochs'],
                 epoch_result['n_total_epochs'],
                 "UNIVERSAL" if epoch_universal else "epoch-specific")
        for ep_name, ep_data in epoch_result['per_epoch'].items():
            log.info("      %s: z=%.2f %s",
                     ep_name, ep_data['z_score'],
                     "MAND" if ep_data['mandatory'] else "")

    return {
        'component': component_name,
        'type': 'biological' if is_biological else 'alien',
        'linearity_r2': lin_r2,
        'nonlinear': nonlinear,
        'temporal_decay_ms': decay_ms,
        'temporal_category': category,
        'temporally_extended': temporally_extended,
        'population_ratio': ratio,
        'population_mi': pop_mi,
        'population_emergent': population_emergent,
        'cross_coupled': cross_coupled,
        'epoch_sensitivity': _convert_numpy(epoch_result) if epoch_result else None,
        'epoch_universal': epoch_universal,
    }


# =========================================================================
# GRAMMAR TABLE
# =========================================================================

def build_grammar_table(bio_results, alien_results):
    """Build the comparison table and compute grammar match score."""
    properties = [
        ('nonlinear', 'Nonlinear'),
        ('temporally_extended', 'Temporally extended'),
        ('population_emergent', 'Population-emergent'),
        ('epoch_universal', 'Epoch-universal'),
    ]

    table = {}
    for prop_key, prop_name in properties:
        bio_vals = [r.get(prop_key) for r in bio_results if r.get(prop_key) is not None]
        alien_vals = [r.get(prop_key) for r in alien_results if r.get(prop_key) is not None]

        bio_majority = (sum(bio_vals) > len(bio_vals) / 2) if bio_vals else None
        alien_majority = (sum(alien_vals) > len(alien_vals) / 2) if alien_vals else None

        match = None
        if bio_majority is not None and alien_majority is not None:
            match = (bio_majority == alien_majority)

        table[prop_name] = {
            'biological': bio_majority,
            'alien': alien_majority,
            'match': match,
            'bio_detail': bio_vals,
            'alien_detail': alien_vals,
        }

    n_testable = sum(1 for v in table.values() if v['match'] is not None)
    n_match = sum(1 for v in table.values() if v['match'] is True)

    return table, n_match, n_testable


def print_grammar_table(table, bio_results, alien_results, n_match, n_testable):
    """Print the final grammar comparison table."""
    log.info("\n" + "=" * 100)
    log.info("COMPUTATIONAL GRAMMAR -- COMPARISON TABLE")
    log.info("=" * 100)

    header = "{:<22} | {:<20}".format("Property", "Bio (sub-2,5,14)")
    for r in alien_results:
        header += " | {:<15}".format("Alien ({})".format(r['component'][:12]))
    header += " | Universal?"
    log.info(header)
    log.info("-" * len(header))

    prop_display = {
        'Nonlinear': ('nonlinear', 'linearity_r2', 'R2'),
        'Temporally extended': ('temporally_extended', 'temporal_decay_ms', 'ms'),
        'Population-emergent': ('population_emergent', 'population_ratio', 'ratio'),
        'Epoch-universal': ('epoch_universal', 'epoch_universal', ''),
    }

    for prop_name, (bool_key, detail_key, unit) in prop_display.items():
        bio_vals = [r.get(bool_key) for r in bio_results if r.get(bool_key) is not None]
        bio_details = [r.get(detail_key, 0) for r in bio_results]
        bio_str = "YES" if any(bio_vals) else "NO"
        if bio_details and unit:
            bio_str += " ({}={:.2f})".format(unit, np.mean([d for d in bio_details if d]))

        line = "{:<22} | {:<20}".format(prop_name, bio_str)

        for r in alien_results:
            val = r.get(bool_key)
            detail = r.get(detail_key, 0)
            if val is None:
                a_str = "N/A"
            elif val:
                a_str = "YES"
            else:
                a_str = "NO"
            if detail and unit:
                a_str += " ({:.2f})".format(detail)
            line += " | {:<15}".format(a_str)

        match = table[prop_name]['match']
        if match is True:
            line += " | YES"
        elif match is False:
            line += " | NO"
        else:
            line += " | N/A"

        log.info(line)

    log.info("-" * 100)
    log.info("\nGRAMMAR MATCH: %d/%d properties shared", n_match, n_testable)

    if n_match == n_testable and n_testable >= 3:
        log.info(">>> OUTCOME 2: Computational grammar of consciousness discovered")
        log.info("    All testable abstract properties are shared between biological")
        log.info("    and alien mandatory variables.")
    elif n_match == 0:
        log.info(">>> OUTCOME 1: No universal grammar -- fully idiosyncratic")
        log.info("    Mandatory variables share no abstract computational properties.")
    else:
        log.info(">>> PARTIAL GRAMMAR: %d universal properties identified", n_match)
        log.info("    Some abstract properties are shared, others are instantiation-specific.")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Computational Grammar Analysis')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--zombie-results-dir',
                        default='results/kyzar/zombie_structure')
    parser.add_argument('--phase34-results-dir', default='results/kyzar')
    parser.add_argument('--output-dir',
                        default='results/kyzar/computational_grammar')
    parser.add_argument('--zombie-subjects', nargs='+', default=['8', '10', '11'])
    parser.add_argument('--bio-subjects', nargs='+', default=['2', '5', '14'])
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("COMPUTATIONAL GRAMMAR ANALYSIS")
    log.info("  Question: Do biological and alien mandatory variables share")
    log.info("            abstract computational properties?")
    log.info("  Bio subjects:    %s", args.bio_subjects)
    log.info("  Zombie subjects: %s", args.zombie_subjects)
    log.info("  Device: %s", args.device)
    log.info("=" * 70)

    t_start = time.time()

    # === BIOLOGICAL MANDATORY VARIABLES ===
    log.info("\n--- BIOLOGICAL MANDATORY VARIABLES ---")
    bio_results = []

    for sub in args.bio_subjects:
        session = 'session_sub{}_ses2'.format(sub)
        log.info("\nLoading bio subject sub-%s...", sub)
        data = load_session_data(
            args.processed_dir, args.model_dir, session, args.device)

        if data is None:
            log.warning("  sub-%s: no data -- skipping", sub)
            continue

        bio_target = get_biological_mandatory_target(data)
        if np.std(bio_target) < 1e-10:
            log.warning("  sub-%s: theta_gamma_pac has zero variance -- skipping", sub)
            continue

        result = analyze_subject(
            data, 'theta_gamma_pac_sub{}'.format(sub),
            bio_target, is_biological=True, device=args.device)
        bio_results.append(result)

    # === ALIEN MANDATORY VARIABLES ===
    log.info("\n--- ALIEN MANDATORY VARIABLES ---")
    alien_results = []

    for sub in args.zombie_subjects:
        session = 'session_sub{}_ses2'.format(sub)
        zombie_path = Path(args.zombie_results_dir) / session / 'zombie_structure.json'

        log.info("\nLoading zombie subject sub-%s...", sub)
        data = load_session_data(
            args.processed_dir, args.model_dir, session, args.device)

        if data is None:
            log.warning("  sub-%s: no data -- skipping", sub)
            continue

        mandatory_names, components = get_alien_mandatory_components(
            zombie_path, data)

        if not mandatory_names:
            log.info("  sub-%s: NO MANDATORY STRUCTURE -- GENUINE ZOMBIE", sub)
            log.info("    Excluded from grammar comparison")
            alien_results.append({
                'component': 'GENUINE_ZOMBIE_sub{}'.format(sub),
                'type': 'alien',
                'nonlinear': None,
                'temporally_extended': None,
                'population_emergent': None,
                'cross_coupled': None,
                'epoch_universal': None,
                'linearity_r2': None,
                'temporal_decay_ms': None,
                'population_ratio': None,
            })
            continue

        # Analyze the first mandatory component
        comp_name = mandatory_names[0]
        if comp_name in components:
            comp_values = components[comp_name]
            result = analyze_subject(
                data, '{}_sub{}'.format(comp_name, sub),
                comp_values, is_biological=False, device=args.device)
            alien_results.append(result)
        else:
            log.warning("  sub-%s: could not reconstruct %s -- skipping", sub, comp_name)

    # === BUILD GRAMMAR TABLE ===
    alien_with_structure = [r for r in alien_results if r.get('nonlinear') is not None]

    table = None
    n_match = 0
    n_testable = 0

    if not bio_results:
        log.error("No biological subjects with data -- cannot build grammar table")
    elif not alien_with_structure:
        log.info("\nAll zombie subjects are GENUINE ZOMBIES -- no alien mandatory structure")
        log.info("Grammar comparison not possible (no alien variables to compare)")
        log.info("This is OUTCOME 3: zombies have no mandatory internal structure at all")
    else:
        table, n_match, n_testable = build_grammar_table(bio_results, alien_with_structure)
        print_grammar_table(table, bio_results, alien_with_structure, n_match, n_testable)

    # Save results
    total_time = time.time() - t_start
    master = {
        'experiment': 'computational_grammar',
        'total_time_min': total_time / 60,
        'bio_subjects': args.bio_subjects,
        'zombie_subjects': args.zombie_subjects,
        'bio_results': [_convert_numpy(r) for r in bio_results],
        'alien_results': [_convert_numpy(r) for r in alien_results],
        'grammar_table': _convert_numpy(table) if table else None,
        'n_match': n_match,
        'n_testable': n_testable,
    }
    with open(output_dir / 'computational_grammar_results.json', 'w') as f:
        json.dump(master, f, indent=2)

    log.info("\nTotal time: %.1f min", total_time / 60)
    log.info("Results: %s", output_dir)


if __name__ == '__main__':
    main()
