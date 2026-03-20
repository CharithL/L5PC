#!/usr/bin/env python3
"""
run_sae_retroactive.py

DESCARTES Circuit 6 — SAE Retroactive Analysis on Zombie Subjects

Zombie subjects (sub-8, sub-10, sub-11) showed zero mandatory variables
under Ridge probing. This script checks whether biological variables are
encoded in polysemantic superposition invisible to linear probes.

Pipeline per subject:
  1. Load Phase 2 hidden states (trained + untrained)
  2. Load bio_targets.npz (18 variables)
  3. Train TopK SAE (expansion_factor=[4,8], k=[10,20])
  4. Ridge delta-R2 probing on SAE sparse features (GroupKFold by trial)
  5. Resample ablation on any variable with SAE delta-R2 > 0.05
  6. Output comparison table: raw Ridge vs SAE Ridge vs resample z-score

Key fix: ALL Ridge probing uses GroupKFold by trial to prevent temporal leakage.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('sae_retroactive')


# -- TopK Sparse Autoencoder (Gao et al. 2024) --

class TopKSAE(nn.Module):
    """TopK SAE: encoder maps h -> sparse features, decoder reconstructs."""

    def __init__(self, input_dim, expansion_factor=4, k=20):
        super().__init__()
        n_features = expansion_factor * input_dim
        self.k = k
        self.input_dim = input_dim
        self.n_features = n_features

        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0)

    def encode(self, x):
        x_centered = x - self.decoder.bias
        pre_act = self.encoder(x_centered)
        topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(-1, topk_idx, torch.relu(topk_vals))
        return sparse

    def forward(self, x):
        sparse = self.encode(x)
        recon = self.decoder(sparse)
        return recon, sparse


def train_sae(hidden_flat, input_dim, expansion_factor=4, k=20,
              lr=1e-3, epochs=100, batch_size=4096, device='cpu'):
    """Train TopK SAE on flattened hidden states."""
    data = torch.tensor(hidden_flat, dtype=torch.float32, device=device)
    N = data.shape[0]

    sae = TopKSAE(input_dim, expansion_factor, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, N, batch_size):
            batch = data[perm[i:i + batch_size]]
            recon, sparse = sae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            norm_loss = ((torch.norm(sae.decoder.weight, dim=0) - 1.0) ** 2).mean()
            loss = recon_loss + 0.01 * norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += recon_loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                sample = data[:min(10000, N)]
                feats = sae.encode(sample)
                alive = (feats > 0).any(dim=0).sum().item()
                recon_var = nn.functional.mse_loss(sae(sample)[0], sample).item()
                total_var = sample.var().item()
                recon_r2 = 1.0 - recon_var / (total_var + 1e-12)
            log.info("    Epoch %d/%d: loss=%.6f, alive=%d/%d, recon_R2=%.4f",
                     epoch + 1, epochs, loss_history[-1], alive,
                     sae.n_features, recon_r2)

    # Final metrics
    sae.eval()
    with torch.no_grad():
        sample = data[:min(50000, N)]
        recon, feats = sae(sample)
        recon_loss = nn.functional.mse_loss(recon, sample).item()
        total_var = sample.var().item()
        recon_r2 = 1.0 - recon_loss / (total_var + 1e-12)
        feat_np = feats.cpu().numpy()
        l0 = (feat_np > 0).sum(axis=1).mean()
        n_dead = int((feat_np.sum(axis=0) == 0).sum())
        n_alive = sae.n_features - n_dead

    metrics = {
        'expansion_factor': expansion_factor,
        'k': k,
        'n_features': sae.n_features,
        'recon_r2': float(recon_r2),
        'mean_l0': float(l0),
        'n_alive': n_alive,
        'n_dead': n_dead,
        'final_loss': float(loss_history[-1]),
    }
    return sae, metrics, loss_history


# -- Ridge delta-R2 with GroupKFold --

def build_trial_groups(hidden_dict, n_trials):
    """Build per-timestep trial group labels for GroupKFold."""
    groups = []
    for ti in range(n_trials):
        key = 'trial_' + str(ti)
        if key not in hidden_dict:
            continue
        arr = hidden_dict[key]
        T = arr.shape[0]
        groups.extend([ti] * T)
    return np.array(groups)


def flatten_trial_data(data_dict, n_trials):
    """Flatten trial-keyed dict to (N_total, dim) array."""
    arrays = []
    for ti in range(n_trials):
        key = 'trial_' + str(ti)
        if key in data_dict:
            arrays.append(data_dict[key])
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def ridge_delta_r2_groupkfold(X_trained, X_untrained, y, groups, n_splits=5):
    """Ridge delta-R2 using GroupKFold by trial -- prevents temporal leakage."""
    if y.std() < 1e-10:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0, 'valid': False}

    y = (y - y.mean()) / (y.std() + 1e-12)
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return {'r2_trained': 0.0, 'r2_untrained': 0.0, 'delta_r2': 0.0, 'valid': False}

    gkf = GroupKFold(n_splits=actual_splits)

    r2_t = float(np.mean(cross_val_score(
        Ridge(alpha=1.0), X_trained, y, cv=gkf, groups=groups, scoring='r2')))
    r2_u = float(np.mean(cross_val_score(
        Ridge(alpha=1.0), X_untrained, y, cv=gkf, groups=groups, scoring='r2')))

    return {
        'r2_trained': r2_t,
        'r2_untrained': r2_u,
        'delta_r2': r2_t - r2_u,
        'valid': True,
    }


# -- Resample Ablation --

class LSTMSurrogate(nn.Module):
    """Must match Phase 2 architecture for checkpoint loading."""
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


def resample_ablation(model, X_data, hidden_dict, encoding_direction,
                      n_trials, n_resample=100, device='cpu'):
    """
    Resample ablation: replace hidden dims along encoding direction with
    values from a different trial. Measures causal necessity.
    """
    model.eval()

    # Collect per-trial hidden states and outputs
    trial_hidden = []
    trial_indices = []
    for ti in range(n_trials):
        key = 'trial_' + str(ti)
        if key not in hidden_dict:
            continue
        trial_hidden.append(hidden_dict[key])
        trial_indices.append(ti)

    n_actual = len(trial_hidden)
    if n_actual < 5:
        return {'z_score': 0.0, 'baseline_mean': 0.0, 'mandatory': False}

    enc_dir = encoding_direction / (np.linalg.norm(encoding_direction) + 1e-10)

    # Baseline: normal forward pass output norm
    baseline_norms = []
    for ti_idx, ti in enumerate(trial_indices):
        key = 'trial_' + str(ti)
        x = torch.tensor(X_data[key], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(x)
        baseline_norms.append(float(pred[0].cpu().numpy().std()))

    # Ablated: swap encoding direction from donor trial
    ablated_scores = []
    for _ in range(n_resample):
        scores = []
        for ti_idx in range(n_actual):
            donor_idx = np.random.randint(0, n_actual)
            while donor_idx == ti_idx and n_actual > 1:
                donor_idx = np.random.randint(0, n_actual)

            h_orig = trial_hidden[ti_idx].copy()
            h_donor = trial_hidden[donor_idx]

            T = h_orig.shape[0]
            T_donor = h_donor.shape[0]
            T_use = min(T, T_donor)

            proj_orig = h_orig[:T_use] @ enc_dir
            proj_donor = h_donor[:T_use] @ enc_dir

            h_ablated = h_orig.copy()
            h_ablated[:T_use] = (h_orig[:T_use]
                                 - np.outer(proj_orig, enc_dir)
                                 + np.outer(proj_donor, enc_dir))

            with torch.no_grad():
                h_t = torch.tensor(h_ablated, dtype=torch.float32).to(device)
                pred_abl = model.output_layer(h_t).cpu().numpy()
            scores.append(float(pred_abl.std()))
        ablated_scores.append(np.mean(scores))

    baseline_mean = float(np.mean(baseline_norms))
    ablated_mean = float(np.mean(ablated_scores))
    ablated_std = float(np.std(ablated_scores)) + 1e-10

    z_score = (ablated_mean - baseline_mean) / ablated_std

    return {
        'z_score': float(z_score),
        'baseline_mean': baseline_mean,
        'ablated_mean': ablated_mean,
        'ablated_std': float(ablated_std),
        'mandatory': z_score < -2.0,
    }


# -- Main Analysis --

def analyze_subject(session_name, processed_dir, model_dir, results_dir,
                    sae_configs, device='cpu'):
    """Run full SAE retroactive analysis for one zombie subject."""
    session_proc = processed_dir / session_name
    session_model = model_dir / session_name

    log.info("=" * 70)
    log.info("Subject: %s", session_name)

    # Find best hidden size with passing output gate
    best_hdim = None
    best_cc = -1
    for hdir in sorted(session_model.glob('lstm_h*')):
        val_path = hdir / 'output_validation.json'
        if not val_path.exists():
            continue
        with open(val_path) as f:
            val = json.load(f)
        if val.get('passed_gate', False) and val['output_cc'] > best_cc:
            best_cc = val['output_cc']
            best_hdim = val['hidden_dim']

    if best_hdim is None:
        log.warning("  No passing model found, trying largest available")
        hdirs = sorted(session_model.glob('lstm_h*'))
        if not hdirs:
            log.error("  No models at all -- skipping")
            return None
        best_hdim = int(hdirs[-1].name.replace('lstm_h', ''))

    model_path = session_model / ('lstm_h' + str(best_hdim))
    log.info("  Using h=%d (CC=%.3f)", best_hdim, best_cc)

    # Load hidden states
    h_trained_path = model_path / 'hidden_trained.npz'
    h_untrained_path = model_path / 'hidden_untrained.npz'
    if not h_trained_path.exists():
        log.error("  hidden_trained.npz not found -- skipping")
        return None

    h_trained_dict = dict(np.load(h_trained_path))
    h_untrained_dict = dict(np.load(h_untrained_path))

    # Load bio targets
    bio_data = dict(np.load(session_proc / 'bio_targets.npz'))
    with open(session_proc / 'bio_variable_names.json') as f:
        bio_names = json.load(f)

    with open(session_proc / 'metadata.json') as f:
        meta = json.load(f)
    n_trials = meta['n_trials']

    # Flatten hidden states
    h_trained_flat = flatten_trial_data(h_trained_dict, n_trials)
    h_untrained_flat = flatten_trial_data(h_untrained_dict, n_trials)
    groups = build_trial_groups(h_trained_dict, n_trials)

    log.info("  Hidden shape: %s, n_trials=%d, n_timesteps=%d",
             h_trained_flat.shape, n_trials, len(groups))

    # Flatten bio targets to match hidden states
    # bio_targets.npz is trial-keyed: trial_0 -> (T, 18), trial_1 -> (T, 18), ...
    # bio_variable_names.json gives the 18 column names
    bio_flat = {}
    for vi, name in enumerate(bio_names):
        pieces = []
        for ti in range(n_trials):
            key = 'trial_' + str(ti)
            if key not in h_trained_dict or key not in bio_data:
                continue
            bio_trial = bio_data[key]  # (T_bio, 18)
            T_h = h_trained_dict[key].shape[0]
            if bio_trial.ndim == 2 and vi < bio_trial.shape[1]:
                col = bio_trial[:, vi]  # (T_bio,)
                if len(col) >= T_h:
                    pieces.append(col[:T_h])
                else:
                    padded = np.zeros(T_h)
                    padded[:len(col)] = col
                    pieces.append(padded)
            elif bio_trial.ndim == 1:
                # Single column -- use directly
                if len(bio_trial) >= T_h:
                    pieces.append(bio_trial[:T_h])
                else:
                    padded = np.zeros(T_h)
                    padded[:len(bio_trial)] = bio_trial
                    pieces.append(padded)
        if pieces:
            bio_flat[name] = np.concatenate(pieces)

    # Verify alignment
    n_total = h_trained_flat.shape[0]
    valid_bio = {k: v for k, v in bio_flat.items() if len(v) == n_total}
    log.info("  Bio variables aligned: %d/%d", len(valid_bio), len(bio_names))

    if not valid_bio:
        log.error("  No bio variables aligned -- skipping")
        return None

    # -- Step 1: Raw Ridge delta-R2 (baseline, with GroupKFold) --
    log.info("  [1/4] Raw Ridge delta-R2 with GroupKFold...")
    raw_results = {}
    for name, y in valid_bio.items():
        res = ridge_delta_r2_groupkfold(h_trained_flat, h_untrained_flat, y, groups)
        raw_results[name] = res
        if abs(res['delta_r2']) > 0.02:
            log.info("    %s: delta_R2=%.3f (trained=%.3f, untrained=%.3f)",
                     name, res['delta_r2'], res['r2_trained'], res['r2_untrained'])

    # -- Step 2: SAE training sweep --
    log.info("  [2/4] Training SAE sweep...")
    sae_results = {}
    hidden_dim = h_trained_flat.shape[1]

    for ef, k_val in sae_configs:
        config_key = 'ef{}_k{}'.format(ef, k_val)
        log.info("    Config: expansion=%d, k=%d", ef, k_val)

        sae_t, met_t, _ = train_sae(
            h_trained_flat, hidden_dim, expansion_factor=ef, k=k_val,
            epochs=100, batch_size=4096, device=device)

        sae_u, met_u, _ = train_sae(
            h_untrained_flat, hidden_dim, expansion_factor=ef, k=k_val,
            epochs=100, batch_size=4096, device=device)

        log.info("    Trained SAE: recon_R2=%.4f, alive=%d/%d, L0=%.1f",
                 met_t['recon_r2'], met_t['n_alive'], met_t['n_features'],
                 met_t['mean_l0'])

        # Extract SAE features
        sae_t.eval()
        sae_u.eval()
        with torch.no_grad():
            ht = torch.tensor(h_trained_flat, dtype=torch.float32, device=device)
            hu = torch.tensor(h_untrained_flat, dtype=torch.float32, device=device)
            feat_trained = sae_t.encode(ht).cpu().numpy()
            feat_untrained = sae_u.encode(hu).cpu().numpy()

        # Ridge delta-R2 on SAE features (GroupKFold)
        sae_probe = {}
        for name, y in valid_bio.items():
            res = ridge_delta_r2_groupkfold(feat_trained, feat_untrained, y, groups)
            sae_probe[name] = res

        # Monosemanticity analysis
        n_bio = len(valid_bio)
        n_feat = feat_trained.shape[1]
        corr_matrix = np.zeros((n_feat, n_bio))
        for fi in range(n_feat):
            feat_col = feat_trained[:, fi]
            if feat_col.std() < 1e-10:
                continue
            for bi, (bname, y) in enumerate(valid_bio.items()):
                if y.std() > 1e-10:
                    corr_matrix[fi, bi] = np.corrcoef(feat_col, y)[0, 1]

        mono_scores = np.zeros(n_feat)
        for fi in range(n_feat):
            abs_corr = np.abs(corr_matrix[fi])
            total = abs_corr.sum()
            if total < 1e-10:
                continue
            probs = abs_corr / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(n_bio)
            mono_scores[fi] = 1.0 - (entropy / max_entropy)

        sae_results[config_key] = {
            'sae_metrics_trained': met_t,
            'sae_metrics_untrained': met_u,
            'probing': {name: _convert(v) for name, v in sae_probe.items()},
            'mean_monosemanticity': float(mono_scores[mono_scores > 0].mean()
                                          if (mono_scores > 0).any() else 0.0),
            'n_monosemantic': int((mono_scores > 0.3).sum()),
        }

        # Print comparison for key variables
        for name in valid_bio:
            raw_dr2 = raw_results[name]['delta_r2']
            sae_dr2 = sae_probe[name]['delta_r2']
            superposed = sae_dr2 > raw_dr2 + 0.05
            marker = " *** SUPERPOSITION" if superposed else ""
            if abs(sae_dr2) > 0.02 or abs(raw_dr2) > 0.02:
                log.info("      %s: raw=%.3f -> SAE=%.3f%s",
                         name, raw_dr2, sae_dr2, marker)

    # -- Step 3: Identify superposition candidates --
    log.info("  [3/4] Checking for superposition...")
    superposition_found = {}
    for config_key, sr in sae_results.items():
        for name in valid_bio:
            raw_dr2 = raw_results[name]['delta_r2']
            sae_dr2 = sr['probing'][name]['delta_r2']
            if sae_dr2 > raw_dr2 + 0.05 and sae_dr2 > 0.05:
                if name not in superposition_found or sae_dr2 > superposition_found[name]['sae_delta_r2']:
                    superposition_found[name] = {
                        'raw_delta_r2': float(raw_dr2),
                        'sae_delta_r2': float(sae_dr2),
                        'config': config_key,
                        'boost': float(sae_dr2 - raw_dr2),
                    }
                    log.info("    SUPERPOSITION: %s -- raw=%.3f, SAE(%s)=%.3f (+%.3f)",
                             name, raw_dr2, config_key, sae_dr2, sae_dr2 - raw_dr2)

    # -- Step 4: Resample ablation on superposition candidates --
    ablation_results = {}
    if superposition_found:
        log.info("  [4/4] Resample ablation on %d superposition candidates...",
                 len(superposition_found))

        val_path = model_path / 'output_validation.json'
        with open(val_path) as f:
            val_info = json.load(f)

        model = LSTMSurrogate(val_info['n_in'], val_info['n_out'], best_hdim).to(device)
        state_dict = torch.load(model_path / 'trained_model.pt',
                                map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        X_data = dict(np.load(session_proc / 'X_trials.npz'))

        for name, sp_info in superposition_found.items():
            log.info("    Ablating %s...", name)
            y = valid_bio[name]
            y_norm = (y - y.mean()) / (y.std() + 1e-10)
            ridge = Ridge(alpha=1.0)
            ridge.fit(h_trained_flat, y_norm)
            enc_dir = ridge.coef_

            abl = resample_ablation(model, X_data, h_trained_dict, enc_dir,
                                    n_trials, n_resample=100, device=device)
            ablation_results[name] = abl
            log.info("    %s: z=%.2f, mandatory=%s",
                     name, abl['z_score'], abl['mandatory'])
    else:
        log.info("  [4/4] No superposition candidates -- skipping ablation")

    # -- Compile results --
    comparison_table = []
    for name in valid_bio:
        row = {
            'variable': name,
            'raw_delta_r2': float(raw_results[name]['delta_r2']),
            'raw_r2_trained': float(raw_results[name]['r2_trained']),
        }
        for config_key, sr in sae_results.items():
            row['sae_' + config_key + '_delta_r2'] = float(sr['probing'][name]['delta_r2'])
            row['sae_' + config_key + '_r2_trained'] = float(sr['probing'][name]['r2_trained'])
        row['superposition_detected'] = name in superposition_found
        if name in superposition_found:
            row['best_sae_boost'] = superposition_found[name]['boost']
        if name in ablation_results:
            row['ablation_z'] = ablation_results[name]['z_score']
            row['ablation_mandatory'] = ablation_results[name]['mandatory']
        comparison_table.append(row)

    # Verdict
    n_superposed = len(superposition_found)
    n_mandatory_sae = sum(1 for v in ablation_results.values() if v['mandatory'])
    genuine_zombie = n_superposed == 0

    result = {
        'session': session_name,
        'hidden_dim': best_hdim,
        'output_cc': float(best_cc),
        'n_trials': n_trials,
        'n_bio_variables': len(valid_bio),
        'sae_configs': [{'expansion_factor': ef, 'k': k} for ef, k in sae_configs],
        'raw_ridge_results': {k: _convert(v) for k, v in raw_results.items()},
        'sae_results': {k: _convert(v) for k, v in sae_results.items()},
        'superposition_found': {k: _convert(v) for k, v in superposition_found.items()},
        'ablation_results': {k: _convert(v) for k, v in ablation_results.items()},
        'comparison_table': comparison_table,
        'verdict': {
            'genuine_zombie': genuine_zombie,
            'n_superposed_variables': n_superposed,
            'n_mandatory_via_sae': n_mandatory_sae,
            'explanation': (
                'GENUINE ZOMBIE: No biological variables found even with SAE decomposition.'
                if genuine_zombie else
                'OVERTURNED: {} variable(s) found in polysemantic superposition, '
                '{} confirmed mandatory via resample ablation.'.format(
                    n_superposed, n_mandatory_sae)
            ),
        },
    }

    # Print comparison table
    log.info("")
    log.info("  -- Comparison Table --")
    log.info("  %-30s  %8s  %s", "Variable", "Raw dR2", "  ".join(
        'SAE({})'.format(c) for c in sae_results.keys()))
    log.info("  " + "-" * 80)
    for row in comparison_table:
        sae_vals = "  ".join(
            '{:+.3f}'.format(row.get('sae_' + c + '_delta_r2', 0))
            for c in sae_results.keys())
        marker = " ***" if row.get('superposition_detected') else ""
        log.info("  %-30s  %+.3f    %s%s",
                 row['variable'], row['raw_delta_r2'], sae_vals, marker)

    log.info("")
    log.info("  VERDICT: %s", result['verdict']['explanation'])

    return result


def _convert(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Kyzar SAE Retroactive Analysis')
    parser.add_argument('--processed-dir', required=True,
                        help='Path to kyzar_processed/')
    parser.add_argument('--model-dir', required=True,
                        help='Path to models/kyzar/')
    parser.add_argument('--results-dir', required=True,
                        help='Path to results/kyzar/')
    parser.add_argument('--subjects', nargs='*', default=['8', '10', '11'],
                        help='Zombie subject IDs (default: 8 10 11)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # SAE configs: (expansion_factor, k)
    sae_configs = [(4, 10), (4, 20), (8, 10), (8, 20)]

    log.info("SAE Retroactive Analysis -- Circuit 6 Zombie Subjects")
    log.info("Subjects: %s", args.subjects)
    log.info("Device: %s", args.device)
    log.info("SAE configs: %s", sae_configs)
    t0 = time.time()

    all_results = {}
    for sub_id in args.subjects:
        session_name = 'session_sub{}_ses2'.format(sub_id)

        if not (processed_dir / session_name).exists():
            log.warning("Processed dir not found: %s -- skipping", session_name)
            continue
        if not (model_dir / session_name).exists():
            log.warning("Model dir not found: %s -- skipping", session_name)
            continue

        result = analyze_subject(
            session_name, processed_dir, model_dir, results_dir,
            sae_configs, device=args.device)

        if result is not None:
            all_results[sub_id] = result
            out_path = results_dir / 'sae_retroactive_sub{}.json'.format(sub_id)
            with open(out_path, 'w') as f:
                json.dump(_convert(result), f, indent=2)
            log.info("  Saved: %s", out_path)

    # Summary
    dt = time.time() - t0
    log.info("")
    log.info("=" * 70)
    log.info("SAE RETROACTIVE ANALYSIS COMPLETE (%.1f min)", dt / 60)
    log.info("=" * 70)

    for sub_id, res in all_results.items():
        v = res['verdict']
        log.info("  sub-%s: %s (CC=%.3f, h=%d)",
                 sub_id,
                 "GENUINE ZOMBIE" if v['genuine_zombie'] else "OVERTURNED",
                 res['output_cc'], res['hidden_dim'])
        if not v['genuine_zombie']:
            for name, sp in res['superposition_found'].items():
                abl = res['ablation_results'].get(name, {})
                z = abl.get('z_score', 'N/A')
                log.info("    %s: raw=%.3f -> SAE=%.3f (z=%s)",
                         name, sp['raw_delta_r2'], sp['sae_delta_r2'],
                         '{:.2f}'.format(z) if isinstance(z, float) else z)

    summary = {
        'analysis': 'SAE Retroactive -- Circuit 6 Zombie Subjects',
        'sae_configs': [{'ef': ef, 'k': k} for ef, k in sae_configs],
        'subjects': {},
        'runtime_minutes': dt / 60,
    }
    for sub_id, res in all_results.items():
        summary['subjects'][sub_id] = {
            'output_cc': res['output_cc'],
            'hidden_dim': res['hidden_dim'],
            'genuine_zombie': res['verdict']['genuine_zombie'],
            'n_superposed': res['verdict']['n_superposed_variables'],
            'n_mandatory_sae': res['verdict']['n_mandatory_via_sae'],
        }

    with open(results_dir / 'sae_retroactive_summary.json', 'w') as f:
        json.dump(_convert(summary), f, indent=2)
    log.info("  Summary: %s", results_dir / 'sae_retroactive_summary.json')


if __name__ == '__main__':
    main()
