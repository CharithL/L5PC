"""
run_50_seeds.py

Phase 3A: 50-Seed Untrained Baseline Distribution

For each circuit (C1-C6) x architecture (LSTM, MLP), generate 50 untrained
models with different random seeds, extract hidden states, and run Ridge
regression R2 for all probe targets.

Produces a distribution of 50 R2_untrained values per variable, establishing
the null distribution of what an untrained network can linearly decode.

This is the core deflation mechanism: any trained R2 that does not exceed
the 97.5th percentile of the untrained distribution cannot be claimed as
evidence of learned representation.

Usage:
    python -m descartes.council_controls.priority3_baseline_variance.run_50_seeds
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Circuit and architecture definitions
# ---------------------------------------------------------------------------

CIRCUITS = {
    "C1": {"desc": "Mouse V1->LM (Allen)", "n_input": 50, "n_output": 50},
    "C2": {"desc": "Mouse S1->M1 (Svoboda)", "n_input": 60, "n_output": 40},
    "C3": {"desc": "Mouse ALM->thalamus (DANDI 000363)", "n_input": 80, "n_output": 60},
    "C4": {"desc": "Human MTL->frontal (DANDI 000576)", "n_input": 100, "n_output": 80},
    "C5": {"desc": "Human limbic->prefrontal (DANDI 000623)", "n_input": 90, "n_output": 70},
    "C6": {"desc": "Human Sternberg WM (DANDI 000469)", "n_input": 120, "n_output": 100},
}

ARCHITECTURES = ["LSTM", "MLP"]

N_SEEDS = 50
N_TIMESTEPS = 2000       # Synthetic sequence length for baseline
WINDOW_BINS = 10          # MLP time-lag window
HIDDEN_SIZE = 128         # Shared hidden dimension
N_CV_FOLDS = 5
RIDGE_ALPHA = 1.0

# Probe targets that may be reported as "mandatory" in the pipeline
PROBE_TARGETS = [
    "firing_rate", "spike_count", "isi_cv", "burst_index",
    "population_sync", "oscillation_power", "phase_coherence",
    "granger_cause", "transfer_entropy", "mutual_info",
]

RESULTS_DIR = Path("results/council_controls/phase3_baseline_variance")


# ---------------------------------------------------------------------------
# Minimal model factories (no training — random weights only)
# ---------------------------------------------------------------------------

class _UntainedLSTM(torch.nn.Module):
    """Minimal LSTM for hidden-state extraction. Never trained."""

    def __init__(self, n_input, n_output, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_input, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, n_output)
        self.hidden_size = hidden_size

    def extract_hidden_states(self, x_seq):
        """x_seq: (T, n_input) numpy array -> (T, hidden_size) numpy."""
        x = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h, _ = self.lstm(x)
        return h.squeeze(0).numpy()


class _UntrainedMLP(torch.nn.Module):
    """Minimal MLP-timelag for hidden-state extraction. Never trained."""

    def __init__(self, n_input, n_output, hidden_size, window_bins):
        super().__init__()
        self.window_bins = window_bins
        flat_in = n_input * window_bins
        self.net = torch.nn.Sequential(
            torch.nn.Linear(flat_in, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_output),
        )
        self.hidden_size = hidden_size

    def extract_hidden_states(self, x_seq):
        """x_seq: (T, n_input) numpy -> (T - window + 1, hidden_size) numpy."""
        T, n_in = x_seq.shape
        if T < self.window_bins:
            return np.zeros((0, self.hidden_size))
        windows = []
        for t in range(self.window_bins - 1, T):
            start = t - self.window_bins + 1
            windows.append(x_seq[start:t + 1].flatten())
        x = torch.tensor(np.array(windows), dtype=torch.float32)
        with torch.no_grad():
            # Extract penultimate hidden layer
            for layer in list(self.net)[:-1]:
                x = layer(x)
        return x.numpy()


def make_untrained_model(arch, n_input, n_output, seed):
    """Create an untrained model with a specific random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arch == "LSTM":
        return _UntainedLSTM(n_input, n_output, HIDDEN_SIZE)
    elif arch == "MLP":
        return _UntrainedMLP(n_input, n_output, HIDDEN_SIZE, WINDOW_BINS)
    else:
        raise ValueError("Unknown architecture: {}".format(arch))


# ---------------------------------------------------------------------------
# Synthetic probe targets (stand-in for real extracted variables)
# ---------------------------------------------------------------------------

def generate_synthetic_targets(T, n_targets, seed):
    """Generate synthetic probe target signals.

    In the full pipeline these come from the actual neural data / simulation.
    Here we use smooth random signals so that the Ridge R2 distribution
    reflects realistic null structure (not pure noise).
    """
    rng = np.random.RandomState(seed + 9999)
    targets = {}
    for i, name in enumerate(PROBE_TARGETS[:n_targets]):
        # Smoothed random walk — mimics slowly-varying neural variables
        raw = rng.randn(T)
        kernel = np.ones(50) / 50
        smoothed = np.convolve(raw, kernel, mode='same')
        targets[name] = smoothed
    return targets


# ---------------------------------------------------------------------------
# Ridge probe R2
# ---------------------------------------------------------------------------

def ridge_r2_cv(H, y, n_folds=N_CV_FOLDS, alpha=RIDGE_ALPHA):
    """Cross-validated Ridge R2 for a single probe target."""
    if H.shape[0] != y.shape[0]:
        min_len = min(H.shape[0], y.shape[0])
        H, y = H[:min_len], y[:min_len]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    for train_idx, test_idx in kf.split(H):
        ridge = Ridge(alpha=alpha)
        ridge.fit(H[train_idx], y[train_idx])
        r2 = ridge.score(H[test_idx], y[test_idx])
        r2_scores.append(r2)
    return float(np.mean(r2_scores))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_seed_sweep(circuits=None, architectures=None, n_seeds=N_SEEDS,
                   output_dir=None):
    """Run the 50-seed untrained baseline sweep.

    Returns:
        dict: {circuit: {arch: {target: [r2_seed0, r2_seed1, ...]}}}
    """
    if circuits is None:
        circuits = CIRCUITS
    if architectures is None:
        architectures = ARCHITECTURES
    if output_dir is None:
        output_dir = RESULTS_DIR

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_combos = len(circuits) * len(architectures)
    combo_idx = 0

    for cid, cinfo in circuits.items():
        all_results[cid] = {}
        for arch in architectures:
            combo_idx += 1
            print("[{}/{}] {} x {} — {} seeds".format(
                combo_idx, total_combos, cid, arch, n_seeds))

            n_in = cinfo["n_input"]
            n_out = cinfo["n_output"]
            target_signals = generate_synthetic_targets(
                N_TIMESTEPS, len(PROBE_TARGETS), seed=hash(cid) & 0xFFFF)

            r2_per_target = {t: [] for t in PROBE_TARGETS}

            for seed in range(n_seeds):
                model = make_untrained_model(arch, n_in, n_out, seed)
                # Random input sequence
                rng = np.random.RandomState(seed + 12345)
                x_seq = rng.randn(N_TIMESTEPS, n_in).astype(np.float32)
                H = model.extract_hidden_states(x_seq)

                for tname, y_full in target_signals.items():
                    y = y_full[:H.shape[0]]
                    r2 = ridge_r2_cv(H, y)
                    r2_per_target[tname].append(r2)

                if (seed + 1) % 10 == 0:
                    print("  seed {}/{} done".format(seed + 1, n_seeds))

            all_results[cid][arch] = {
                t: vals for t, vals in r2_per_target.items()
            }

    # Save raw results
    out_file = out_path / "raw_50seed_r2.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRaw R2 distributions saved: {}".format(out_file))

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_seed_sweep()
    elapsed = time.time() - t0
    print("Total elapsed: {:.1f}s".format(elapsed))
