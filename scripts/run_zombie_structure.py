#!/usr/bin/env python3
"""
run_zombie_structure.py

DESCARTES Circuit 6 -- Probe Zombie Computational Structure

For zombie subjects (sub-8, sub-10, sub-11), discover what computation
they DO perform, even though it doesn't align with biological variables.

Pipeline:
  Step 1: Discover candidate internal variables (unsupervised)
    - PCA top 10 components
    - ICA (FastICA) 10 independent components
    - SAE alive features (from existing SAE retroactive analysis)
    - K-means clustering of hidden state trajectories

  Step 2: Resample ablation on discovered components
    - For each PCA/ICA/SAE component, does ablating it degrade output?
    - If yes: mandatory intermediate of alien computation
    - If ALL no: fully distributed, no separable mandatory structure

  Step 3: Characterize mandatory alien components
    - Correlate with input/output statistics
    - Correlate with task epoch and behavioral variables
    - Visualize activation patterns

  Step 4: Cross-reference with biological variables
    - If mandatory PCA component correlates with theta_gamma_pac,
      sub-8 DOES encode PAC in a subspace Ridge missed
    - If nothing biological correlates: genuinely alien computation

Usage:
  python scripts/run_zombie_structure.py \
    --processed-dir data/kyzar_processed \
    --model-dir models/kyzar \
    --results-dir results/kyzar/zombie_structure \
    --subjects 8 10 11 \
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
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('zombie_structure')


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
# MODEL (must match Phase 2 architecture)
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
# TopK SAE
# =========================================================================

class TopKSAE(nn.Module):
    def __init__(self, input_dim, expansion_factor=4, k=20):
        super().__init__()
        n_features = expansion_factor * input_dim
        self.k = k
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

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


def train_sae(hidden_flat, input_dim, expansion_factor=4, k=10,
              lr=1e-3, epochs=80, batch_size=4096, device='cpu'):
    data = torch.tensor(hidden_flat, dtype=torch.float32, device=device)
    N = data.shape[0]
    sae = TopKSAE(input_dim, expansion_factor, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

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

    sae.eval_mode = True
    with torch.no_grad():
        sample = data[:min(50000, N)]
        feats = sae.encode(sample).cpu().numpy()
        alive_mask = (feats > 0).any(axis=0)
        n_alive = int(alive_mask.sum())

    return sae, n_alive, alive_mask


# =========================================================================
# STEP 1: Discover candidate internal variables
# =========================================================================

def discover_pca_components(H, n_components=10):
    log.info("  PCA: extracting %d components from (%d, %d)...",
             n_components, H.shape[0], H.shape[1])
    n_comp = min(n_components, H.shape[1], H.shape[0])
    pca = PCA(n_components=n_comp)
    projections = pca.fit_transform(H)
    log.info("    Explained variance: %s",
             ', '.join('{:.3f}'.format(v) for v in pca.explained_variance_ratio_[:5]))
    log.info("    Cumulative top-%d: %.3f", n_comp,
             sum(pca.explained_variance_ratio_))
    return projections, pca


def discover_ica_components(H, n_components=10):
    log.info("  ICA: extracting %d independent components...", n_components)
    n_comp = min(n_components, H.shape[1])
    try:
        ica = FastICA(n_components=n_comp, max_iter=500, random_state=42)
        projections = ica.fit_transform(H)
        log.info("    ICA converged: %d components", n_comp)
        return projections, ica
    except Exception as e:
        log.warning("    ICA failed: %s -- falling back to PCA", str(e))
        return discover_pca_components(H, n_components)


def discover_sae_features(H, hidden_dim, device='cpu'):
    log.info("  SAE: training TopK SAE (expansion=4, k=10)...")
    sae, n_alive, alive_mask = train_sae(
        H.astype(np.float32), hidden_dim,
        expansion_factor=4, k=10, epochs=80, device=device)

    sae.eval()
    with torch.no_grad():
        data_t = torch.tensor(H.astype(np.float32), device=device)
        batch_size = 8192
        all_feats = []
        for i in range(0, len(data_t), batch_size):
            batch = data_t[i:i+batch_size]
            feats = sae.encode(batch).cpu().numpy()
            all_feats.append(feats)
        all_features = np.concatenate(all_feats, axis=0)

    alive_features = all_features[:, alive_mask]
    log.info("    SAE: %d alive features out of %d total",
             n_alive, sae.n_features)
    return alive_features, alive_mask, sae


def discover_kmeans_states(H, n_clusters=8):
    log.info("  K-means: finding %d computational states...", n_clusters)
    n_sub = min(50000, len(H))
    idx = np.random.choice(len(H), n_sub, replace=False)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(H[idx])
    labels = kmeans.predict(H)
    distances = kmeans.transform(H)
    log.info("    Cluster sizes: %s",
             ', '.join(str(int(s)) for s in np.bincount(labels)))
    return labels, distances, kmeans


# =========================================================================
# STEP 2: Resample ablation on discovered components
# =========================================================================

def ablation_on_component(H_trained, component_values, groups,
                          model_output_layer, hidden_dim,
                          n_resamples=15, device='cpu'):
    """Resample ablation on a single component direction.

    Replaces the component's projection with values from random donor
    trials and measures output degradation vs random direction baseline.
    """
    with torch.no_grad():
        H_t = torch.tensor(H_trained.astype(np.float32), device=device)
        baseline_output = model_output_layer(H_t).cpu().numpy()

    baseline_var = float(np.var(baseline_output))
    if baseline_var < 1e-10:
        return {'targeted_degradation': 0.0, 'random_degradation': 0.0,
                'z_score': 0.0, 'mandatory': False}

    # Find direction in hidden space via correlation
    corrs = np.array([np.corrcoef(H_trained[:, d], component_values)[0, 1]
                      if np.std(H_trained[:, d]) > 1e-10 else 0.0
                      for d in range(hidden_dim)])
    corrs = np.nan_to_num(corrs, nan=0.0)
    direction = corrs / (np.linalg.norm(corrs) + 1e-10)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # Targeted ablation: swap component direction from donor trial
    targeted_degradations = []
    rng = np.random.default_rng(42)
    for _ in range(n_resamples):
        H_abl = H_trained.copy()
        for g in unique_groups:
            mask = groups == g
            donor = unique_groups[rng.integers(n_groups)]
            while donor == g and n_groups > 1:
                donor = unique_groups[rng.integers(n_groups)]
            donor_mask = groups == donor

            proj_orig = H_abl[mask] @ direction
            h_donor = H_trained[donor_mask]
            n_use = min(np.sum(mask), np.sum(donor_mask))
            proj_donor = h_donor[:n_use] @ direction
            n_replace = min(len(proj_orig), len(proj_donor))

            replacement = (H_abl[mask][:n_replace]
                           - np.outer(proj_orig[:n_replace], direction)
                           + np.outer(proj_donor[:n_replace], direction))
            H_abl[mask] = np.concatenate([replacement, H_abl[mask][n_replace:]], axis=0)

        with torch.no_grad():
            H_abl_t = torch.tensor(H_abl.astype(np.float32), device=device)
            abl_output = model_output_layer(H_abl_t).cpu().numpy()
        mse = float(np.mean((abl_output - baseline_output) ** 2))
        targeted_degradations.append(mse / (baseline_var + 1e-10))

    # Random direction baseline
    random_degradations = []
    for _ in range(n_resamples):
        rand_dir = rng.standard_normal(hidden_dim)
        rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)

        H_rand = H_trained.copy()
        proj = H_rand @ rand_dir
        H_rand -= np.outer(proj, rand_dir)

        with torch.no_grad():
            H_r_t = torch.tensor(H_rand.astype(np.float32), device=device)
            rand_output = model_output_layer(H_r_t).cpu().numpy()
        mse = float(np.mean((rand_output - baseline_output) ** 2))
        random_degradations.append(mse / (baseline_var + 1e-10))

    targ_mean = float(np.mean(targeted_degradations))
    rand_mean = float(np.mean(random_degradations))
    rand_std = float(np.std(random_degradations)) + 1e-10
    z_score = (targ_mean - rand_mean) / rand_std

    return {
        'targeted_degradation': targ_mean,
        'random_degradation': rand_mean,
        'z_score': z_score,
        'mandatory': z_score > 2.0,
    }


# =========================================================================
# STEP 3: Characterize mandatory components
# =========================================================================

def characterize_component(component_values, X_flat, Y_flat, epoch_flat, groups):
    result = {}

    input_mean = np.mean(X_flat, axis=1)
    input_std = np.std(X_flat, axis=1)
    input_max = np.max(X_flat, axis=1)

    for name, vals in [('input_mean_rate', input_mean),
                       ('input_std', input_std),
                       ('input_max', input_max)]:
        if np.std(vals) > 1e-10 and np.std(component_values) > 1e-10:
            r = float(np.corrcoef(component_values, vals)[0, 1])
            result['corr_' + name] = r if not np.isnan(r) else 0.0
        else:
            result['corr_' + name] = 0.0

    output_mean = np.mean(Y_flat, axis=1)
    output_std = np.std(Y_flat, axis=1)
    if np.std(output_mean) > 1e-10:
        r = float(np.corrcoef(component_values, output_mean)[0, 1])
        result['corr_output_mean'] = r if not np.isnan(r) else 0.0
    else:
        result['corr_output_mean'] = 0.0
    if np.std(output_std) > 1e-10:
        r = float(np.corrcoef(component_values, output_std)[0, 1])
        result['corr_output_std'] = r if not np.isnan(r) else 0.0
    else:
        result['corr_output_std'] = 0.0

    if epoch_flat is not None and len(epoch_flat) == len(component_values):
        epoch_f = epoch_flat.astype(float)
        if np.std(epoch_f) > 1e-10:
            r = float(np.corrcoef(component_values, epoch_f)[0, 1])
            result['corr_epoch'] = r if not np.isnan(r) else 0.0
        else:
            result['corr_epoch'] = 0.0

    # Temporal pattern classification
    n_check = min(1000, len(component_values))
    diffs = np.diff(component_values[:n_check])
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    result['sign_changes_per_100'] = float(sign_changes / (n_check / 100.0))

    autocorr_lag1 = float(np.corrcoef(
        component_values[:-1], component_values[1:])[0, 1])
    result['autocorrelation_lag1'] = autocorr_lag1 if not np.isnan(autocorr_lag1) else 0.0

    if result['autocorrelation_lag1'] > 0.95:
        result['pattern_type'] = 'slow_drift_or_ramp'
    elif result['sign_changes_per_100'] > 30:
        result['pattern_type'] = 'oscillatory'
    elif result['sign_changes_per_100'] < 5:
        result['pattern_type'] = 'step_like'
    else:
        result['pattern_type'] = 'mixed'

    return result


# =========================================================================
# STEP 4: Cross-reference with biological variables
# =========================================================================

def cross_reference_biology(component_values, bio_flat, bio_names):
    correlations = {}
    for name in bio_names:
        if name not in bio_flat:
            continue
        bio_vals = bio_flat[name]
        n = min(len(component_values), len(bio_vals))
        if n < 100:
            continue
        cv = component_values[:n]
        bv = bio_vals[:n]
        if np.std(cv) > 1e-10 and np.std(bv) > 1e-10:
            r = float(np.corrcoef(cv, bv)[0, 1])
            correlations[name] = r if not np.isnan(r) else 0.0
        else:
            correlations[name] = 0.0

    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_match = sorted_corrs[0] if sorted_corrs else ('none', 0.0)

    return {
        'all_correlations': correlations,
        'top_bio_match': top_match[0],
        'top_bio_r': top_match[1],
        'any_strong_match': abs(top_match[1]) > 0.5,
    }


# =========================================================================
# PER-SUBJECT ANALYSIS
# =========================================================================

def analyze_subject(session_name, processed_dir, model_dir, results_dir,
                    device='cpu'):
    session_proc = Path(processed_dir) / session_name
    session_model = Path(model_dir) / session_name

    log.info("\n" + "=" * 70)
    log.info("ZOMBIE STRUCTURE ANALYSIS: %s", session_name)
    log.info("=" * 70)

    # --- Find best model ---
    best_hdim = None
    best_cc = -1
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
            log.error("  No models found -- skipping")
            return None
        best_hdim = int(hdirs[-1].name.replace('lstm_h', ''))
        best_cc = 0.0

    model_path = session_model / ('lstm_h' + str(best_hdim))
    log.info("  Model: h=%d (CC=%.3f)", best_hdim, best_cc)

    # --- Load hidden states ---
    h_trained_path = model_path / 'hidden_trained.npz'
    if not h_trained_path.exists():
        log.error("  hidden_trained.npz not found -- skipping")
        return None

    h_trained_dict = dict(np.load(h_trained_path))

    # --- Load metadata ---
    with open(session_proc / 'metadata.json') as f:
        meta = json.load(f)
    n_trials = meta['n_trials']

    # --- Flatten hidden states ---
    h_arrays, groups_list = [], []
    for ti in range(n_trials):
        key = 'trial_' + str(ti)
        if key in h_trained_dict:
            arr = h_trained_dict[key]
            h_arrays.append(arr)
            groups_list.extend([ti] * arr.shape[0])

    if not h_arrays:
        log.error("  No hidden state data -- skipping")
        return None

    H = np.concatenate(h_arrays, axis=0).astype(np.float64)
    groups = np.array(groups_list)
    hidden_dim = H.shape[1]
    log.info("  Hidden states: %s, %d trials", H.shape, len(np.unique(groups)))

    # --- Load X, Y data for characterization ---
    X_data = dict(np.load(session_proc / 'X_trials.npz'))
    Y_data = dict(np.load(session_proc / 'Y_trials.npz'))

    X_arrays, Y_arrays, epoch_arrays = [], [], []
    E_data_path = session_proc / 'epoch_masks.npz'
    E_data = dict(np.load(E_data_path)) if E_data_path.exists() else {}

    for ti in range(n_trials):
        key = 'trial_' + str(ti)
        if key in h_trained_dict and key in X_data:
            T_h = h_trained_dict[key].shape[0]
            x = X_data[key][:T_h]
            y = Y_data[key][:T_h] if key in Y_data else np.zeros((T_h, 1))
            e = E_data[key][:T_h] if key in E_data else np.zeros(T_h, dtype=int)
            X_arrays.append(x)
            Y_arrays.append(y)
            epoch_arrays.append(e)

    X_flat = np.concatenate(X_arrays, axis=0)
    Y_flat = np.concatenate(Y_arrays, axis=0)
    epoch_flat = np.concatenate(epoch_arrays, axis=0)

    # --- Load bio targets ---
    bio_flat = {}
    bio_names = []
    bio_path = session_proc / 'bio_targets.npz'
    names_path = session_proc / 'bio_variable_names.json'
    if bio_path.exists() and names_path.exists():
        bio_data = dict(np.load(bio_path))
        with open(names_path) as f:
            bio_names = json.load(f)
        for vi, name in enumerate(bio_names):
            pieces = []
            for ti in range(n_trials):
                key = 'trial_' + str(ti)
                if key in h_trained_dict and key in bio_data:
                    T_h = h_trained_dict[key].shape[0]
                    bt = bio_data[key]
                    if bt.ndim == 2 and vi < bt.shape[1]:
                        pieces.append(bt[:T_h, vi])
                    elif bt.ndim == 1:
                        pieces.append(bt[:T_h])
            if pieces:
                bio_flat[name] = np.concatenate(pieces)
    else:
        log.warning("  Bio targets not found -- Step 4 will be limited")

    # --- Load LSTM model for output layer ---
    checkpoint_path = model_path / 'trained_model.pt'
    output_layer = None
    if checkpoint_path.exists():
        n_in = X_flat.shape[1]
        n_out = Y_flat.shape[1]
        model = LSTMSurrogate(n_in, n_out, best_hdim).to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        output_layer = model.output_layer
    else:
        log.warning("  trained_model.pt not found -- skipping ablation")

    # =====================================================================
    # STEP 1: DISCOVER CANDIDATE VARIABLES
    # =====================================================================
    log.info("\n--- STEP 1: DISCOVER CANDIDATE VARIABLES ---")

    pca_proj, pca_model = discover_pca_components(H, n_components=10)
    ica_proj, ica_model = discover_ica_components(H, n_components=10)
    sae_features, sae_alive_mask, sae_model = discover_sae_features(
        H, hidden_dim, device=device)
    km_labels, km_distances, km_model = discover_kmeans_states(H, n_clusters=8)

    # Collect all candidates
    candidates = {}
    for i in range(pca_proj.shape[1]):
        candidates['PCA_{}'.format(i)] = pca_proj[:, i]
    for i in range(ica_proj.shape[1]):
        candidates['ICA_{}'.format(i)] = ica_proj[:, i]
    for i in range(sae_features.shape[1]):
        candidates['SAE_{}'.format(i)] = sae_features[:, i]
    for i in range(km_distances.shape[1]):
        candidates['KM_dist_{}'.format(i)] = km_distances[:, i]

    log.info("  Total candidates: %d (PCA:%d ICA:%d SAE:%d KM:%d)",
             len(candidates), pca_proj.shape[1], ica_proj.shape[1],
             sae_features.shape[1], km_distances.shape[1])

    # =====================================================================
    # STEP 2: RESAMPLE ABLATION
    # =====================================================================
    log.info("\n--- STEP 2: RESAMPLE ABLATION ---")

    ablation_results = {}
    mandatory_components = []

    if output_layer is not None:
        # Ablate PCA + ICA components
        for name in sorted(candidates.keys()):
            if not (name.startswith('PCA_') or name.startswith('ICA_')):
                continue
            comp = candidates[name]
            if np.std(comp) < 1e-10:
                continue
            abl = ablation_on_component(
                H, comp, groups, output_layer, hidden_dim,
                n_resamples=15, device=device)
            ablation_results[name] = abl
            status = "MANDATORY" if abl['mandatory'] else "byproduct"
            log.info("    %s: z=%.2f (%s) targ=%.4f rand=%.4f",
                     name, abl['z_score'], status,
                     abl['targeted_degradation'], abl['random_degradation'])
            if abl['mandatory']:
                mandatory_components.append(name)

        # Top 10 SAE features by activation strength
        sae_strength = np.mean(np.abs(sae_features), axis=0)
        top_sae_idx = np.argsort(sae_strength)[-min(10, len(sae_strength)):]
        for rank, idx in enumerate(reversed(top_sae_idx)):
            name = 'SAE_{}'.format(idx)
            comp = sae_features[:, idx]
            if np.std(comp) < 1e-10:
                continue
            abl = ablation_on_component(
                H, comp, groups, output_layer, hidden_dim,
                n_resamples=15, device=device)
            ablation_results[name] = abl
            status = "MANDATORY" if abl['mandatory'] else "byproduct"
            log.info("    %s (rank %d): z=%.2f (%s)",
                     name, rank, abl['z_score'], status)
            if abl['mandatory']:
                mandatory_components.append(name)

    n_mandatory = len(mandatory_components)
    log.info("\n  MANDATORY: %d -- %s",
             n_mandatory,
             ', '.join(mandatory_components) if mandatory_components else 'NONE')

    if n_mandatory == 0:
        log.info("  *** FULLY DISTRIBUTED: no separable mandatory structure ***")
    else:
        log.info("  *** %d mandatory intermediates of alien computation ***", n_mandatory)

    # =====================================================================
    # STEP 3: CHARACTERIZE MANDATORY COMPONENTS
    # =====================================================================
    log.info("\n--- STEP 3: CHARACTERIZE ---")

    characterization = {}
    for name in mandatory_components:
        comp = candidates[name]
        char = characterize_component(comp, X_flat, Y_flat, epoch_flat, groups)
        characterization[name] = char
        log.info("  %s: pattern=%s autocorr=%.3f",
                 name, char['pattern_type'], char['autocorrelation_lag1'])
        log.info("    input_rate r=%.3f, output r=%.3f, epoch r=%.3f",
                 char.get('corr_input_mean_rate', 0),
                 char.get('corr_output_mean', 0),
                 char.get('corr_epoch', 0))

    # =====================================================================
    # STEP 4: CROSS-REFERENCE WITH BIOLOGY
    # =====================================================================
    log.info("\n--- STEP 4: CROSS-REFERENCE BIOLOGY ---")

    biology_crossref = {}
    for name in mandatory_components:
        comp = candidates[name]
        xref = cross_reference_biology(comp, bio_flat, bio_names)
        biology_crossref[name] = xref
        log.info("  %s: top=%s (r=%.3f) %s",
                 name, xref['top_bio_match'], xref['top_bio_r'],
                 "BIO_MATCH" if xref['any_strong_match'] else "ALIEN")
        sorted_corrs = sorted(xref['all_correlations'].items(),
                              key=lambda x: abs(x[1]), reverse=True)[:5]
        for bname, r in sorted_corrs:
            log.info("      %s: r=%.3f", bname, r)

    # Cross-reference ALL PCA components
    log.info("\n  All PCA vs biology:")
    pca_biology = {}
    for i in range(pca_proj.shape[1]):
        name = 'PCA_{}'.format(i)
        comp = pca_proj[:, i]
        xref = cross_reference_biology(comp, bio_flat, bio_names)
        pca_biology[name] = xref
        log.info("    %s: top=%s (r=%.3f) var=%.3f %s",
                 name, xref['top_bio_match'], xref['top_bio_r'],
                 pca_model.explained_variance_ratio_[i],
                 "***" if xref['any_strong_match'] else "")

    # =====================================================================
    # VERDICT
    # =====================================================================
    any_bio_match = any(
        biology_crossref.get(n, {}).get('any_strong_match', False)
        for n in mandatory_components
    )

    if n_mandatory == 0:
        verdict = 'FULLY_DISTRIBUTED_ZOMBIE'
        interpretation = (
            'No separable mandatory internal structure. '
            'Computation is fully distributed -- zombie relative to '
            'its own computation, not just biology.'
        )
    elif any_bio_match:
        verdict = 'BIOLOGICAL_IN_SUBSPACE'
        interpretation = (
            'Mandatory components correlate with biological variables. '
            'Biology IS encoded but in a rotated subspace Ridge missed. '
            'Not a true zombie.'
        )
    else:
        verdict = 'GENUINELY_ALIEN'
        interpretation = (
            'Mandatory internal structure exists but correlates with '
            'no biological variable. Genuinely alien computation -- '
            'the most interesting result.'
        )

    log.info("\n  === VERDICT: %s ===", verdict)
    log.info("  %s", interpretation)

    result = {
        'session': session_name,
        'hidden_dim': best_hdim,
        'output_cc': best_cc,
        'n_candidates': len(candidates),
        'n_mandatory': n_mandatory,
        'mandatory_components': mandatory_components,
        'verdict': verdict,
        'interpretation': interpretation,
        'pca_explained_variance': pca_model.explained_variance_ratio_.tolist(),
        'ablation': ablation_results,
        'characterization': characterization,
        'biology_crossref': biology_crossref,
        'pca_biology': pca_biology,
        'kmeans_cluster_sizes': np.bincount(km_labels).tolist(),
        'sae_n_alive': sae_features.shape[1],
    }

    out_dir = Path(results_dir) / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'zombie_structure.json', 'w') as f:
        json.dump(_convert_numpy(result), f, indent=2)

    return result


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Zombie Computational Structure Analysis')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--results-dir', default='results/kyzar/zombie_structure')
    parser.add_argument('--subjects', nargs='+', default=['8', '10', '11'])
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("ZOMBIE COMPUTATIONAL STRUCTURE ANALYSIS")
    log.info("  Question: What do zombie subjects compute internally?")
    log.info("  Subjects: %s", args.subjects)
    log.info("  Device: %s", args.device)
    log.info("=" * 70)

    t_start = time.time()
    all_results = []

    for sub in args.subjects:
        session_name = 'session_sub{}_ses2'.format(sub)
        r = analyze_subject(
            session_name, args.processed_dir, args.model_dir,
            args.results_dir, device=args.device)
        if r is not None:
            all_results.append(r)

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("ZOMBIE STRUCTURE SUMMARY")
    log.info("=" * 70)
    log.info("%-20s | %-6s | %-8s | %-25s | %s",
             "Subject", "CC", "Mand", "Verdict", "Top Bio Match")
    log.info("-" * 90)

    for r in all_results:
        top_bio = 'N/A'
        for mc in r['mandatory_components']:
            xr = r['biology_crossref'].get(mc, {})
            if xr.get('top_bio_match'):
                top_bio = '{} (r={:.2f})'.format(
                    xr['top_bio_match'], xr['top_bio_r'])
                break
        log.info("%-20s | %-6.3f | %-8d | %-25s | %s",
                 r['session'], r['output_cc'], r['n_mandatory'],
                 r['verdict'], top_bio)

    total_time = time.time() - t_start
    master = {
        'experiment': 'zombie_computational_structure',
        'total_time_min': total_time / 60,
        'subjects': [_convert_numpy(r) for r in all_results],
    }
    out_path = Path(args.results_dir) / 'zombie_structure_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(master, f, indent=2)

    log.info("\nTotal time: %.1f min", total_time / 60)
    log.info("Results: %s", args.results_dir)


if __name__ == '__main__':
    main()
