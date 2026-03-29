#!/usr/bin/env python3
"""
Task-Dependence Test v2: Does firing_rate_input disappear when the output
format no longer contains firing rates?

Uses sub-5 from Kyzar dataset. Same input spike trains for all three tasks.
Only the prediction target changes.

Task A:  Spike prediction (output = 29 neurons) — CONTROL
Task B2: Output population mean rate (scalar per timestep — learnable,
         but a scalar summary, not a spike train)
Task C2: Temporal epoch prediction (5-class: fixation/encoding/maintenance/
         probe/response — changes within trials, decodable from input dynamics)

v1 Tasks B/C failed because constant per-trial labels (accuracy, load)
gave no per-timestep gradient signal. v2 tasks vary within trials.

Prediction:
  firing_rate_input is MANDATORY in Task A (tautological: output contains rates)
  firing_rate_input may SURVIVE Task B2 (output mean rate still contains rate info)
    but at REDUCED R2 (scalar vs 29-neuron spike train)
  firing_rate_input DROPS OUT of Task C2 (epoch has no rate information)
  theta_gamma_pac SURVIVES all tasks (reflects temporal world structure)

CRITICAL: All probing functions imported from run_synthetic_reality_v3_iaaft.py.
No reimplementation of iAAFT, ablation, or bio variable computation.

Usage:
  python scripts/run_task_dependence_test.py \
    --processed-dir data/kyzar_processed \
    --output-dir results/task_dependence \
    --source-subject 5 \
    --device cuda
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

# CRITICAL: Import existing pipeline functions — DO NOT REIMPLEMENT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_synthetic_reality_v3_iaaft import (
    LSTMSurrogate,
    compute_bio_variables,
    iaaft_screen,
    resample_ablation,
    load_source_data,
    gap_cv_split,
    trials_to_npz,
    collate_trials,
    train_model,
    extract_hidden_states,
    _convert_numpy,
    PROBE_VARIABLES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('task_dependence')


# =========================================================================
# TASK-SPECIFIC SEEDS (different initialization per task)
# =========================================================================
TASK_SEEDS = {
    'task_a': 42,
    'task_b2': 137,
    'task_c2': 271,
}


# =========================================================================
# STEP 1: Load data and extract behavioral labels
# =========================================================================

def load_sub5_data(processed_dir, subject='5'):
    """Load sub-5 trials from Kyzar processed data.

    Returns:
        trials: list of dicts with X, Y, epoch, meta (same as v3)
        meta: full session metadata
    """
    trials, meta = load_source_data(processed_dir, subject)
    return trials, meta


# =========================================================================
# STEP 2: Create time-varying targets for Tasks B2 and C2
# =========================================================================

def prepare_task_targets(trials):
    """Create three sets of training targets from the same input data.

    Task A:  output spike trains (29 neurons, unchanged)
    Task B2: output population mean firing rate (scalar per timestep)
             — learnable because it's a simplified version of Task A
             — still contains rate information but as a scalar, not spike train
    Task C2: temporal epoch identity (5-class one-hot per timestep)
             — changes WITHIN each trial (fixation->encoding->maintenance->probe->response)
             — does NOT contain firing rate information
             — forces LSTM to learn temporal dynamics

    Returns dict of task_name -> list of (T, n_output) arrays
    """
    tasks = {}

    # Task A: existing spike targets (unchanged)
    tasks['task_a'] = [t['Y'] for t in trials]

    # Task B2: output population mean rate per timestep
    # This is mean(Y, axis=1) — a scalar time series the LSTM can learn
    tasks['task_b2'] = []
    for t in trials:
        Y = t['Y']
        mean_rate = np.mean(Y, axis=1, keepdims=True)  # (T, 1)
        # Epsilon-safe scaling
        mr_mean = np.mean(mean_rate)
        mr_std = np.std(mean_rate) + 1e-8
        mean_rate_scaled = (mean_rate - mr_mean) / mr_std
        tasks['task_b2'].append(mean_rate_scaled.astype(np.float32))

    # Task C2: temporal epoch identity as one-hot per timestep
    # epoch masks encode: 0=fixation, 1=encoding, 2=maintenance, 3=probe, 4=response
    n_epochs = 5
    tasks['task_c2'] = []
    epoch_counts = np.zeros(n_epochs, dtype=int)
    for t in trials:
        epoch = t['epoch']
        T = len(epoch)
        one_hot = np.zeros((T, n_epochs), dtype=np.float32)
        for ei in range(n_epochs):
            one_hot[epoch == ei, ei] = 1.0
        tasks['task_c2'].append(one_hot)
        for ei in range(n_epochs):
            epoch_counts[ei] += int(np.sum(epoch == ei))

    # Diagnostics
    log.info("Task targets prepared:")
    log.info("  Task A  output dim: %d (spike trains — 29 neurons)", tasks['task_a'][0].shape[1])

    b2_example = tasks['task_b2'][0]
    log.info("  Task B2 output dim: %d (output mean rate, per-timestep scalar)", b2_example.shape[1])
    log.info("    B2 range: [%.3f, %.3f], std=%.3f (z-scored)",
             b2_example.min(), b2_example.max(), b2_example.std())

    log.info("  Task C2 output dim: %d (epoch one-hot: fix/enc/maint/probe/resp)", n_epochs)
    log.info("    C2 epoch distribution: %s",
             dict(zip(['fix', 'enc', 'maint', 'probe', 'resp'], epoch_counts)))

    return tasks


# =========================================================================
# STEP 3: Train LSTM for each task (uses existing infrastructure)
# =========================================================================

def train_task_lstm(trials, target_list, task_name, hidden_dim=64,
                    device='cpu', max_epochs=300, patience=20):
    """Train an LSTM for one task using existing infrastructure.

    The ONLY thing that changes between tasks is the Y target.
    X (input) stays the same. Architecture stays the same.

    For Tasks B/C: MSE loss on the constant per-trial label works fine —
    the LSTM learns to output the correct constant from evolving input.

    Returns: model, H_trained, trial_groups, cc, n_epochs
    """
    seed = TASK_SEEDS[task_name]
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_in = trials[0]['X'].shape[1]
    n_out = target_list[0].shape[1]
    n_trials = len(trials)

    log.info("  Training %s: n_in=%d, n_out=%d, h=%d, seed=%d",
             task_name, n_in, n_out, hidden_dim, seed)

    # Build X/Y data dicts in the format train_model expects
    X_data = {}
    Y_data = {}
    for i in range(n_trials):
        key = 'trial_{}'.format(i)
        X_data[key] = trials[i]['X'].astype(np.float32)
        Y_data[key] = target_list[i].astype(np.float32)

    train_idx, test_idx = gap_cv_split(n_trials)

    # Train
    model = LSTMSurrogate(n_in, n_out, hidden_dim).to(device)
    model, cc, n_epochs = train_model(
        model, X_data, Y_data, train_idx, test_idx,
        max_epochs=max_epochs, patience=patience, device=device)

    log.info("  %s: CC=%.3f (%d epochs)", task_name, cc, n_epochs)

    # Quality gate — CC must be POSITIVE and above threshold
    # Negative CC means the model learned anti-correlated output (garbage)
    if task_name == 'task_a':
        threshold = 0.30
    elif task_name == 'task_b2':
        threshold = 0.15  # Scalar prediction should be learnable
    else:  # task_c2
        threshold = 0.10  # 5-class epoch prediction

    passed = cc > threshold  # Must be positive AND above threshold

    log.info("  Quality gate: CC=%.3f %s (threshold: >%.2f, must be positive)",
             cc, "PASS" if passed else "FAIL", threshold)

    # Extract hidden states
    h_dict = extract_hidden_states(model, X_data, n_trials, device)

    # Concatenate hidden states and build trial groups
    h_all = []
    groups_all = []
    for ti in range(n_trials):
        key = 'trial_{}'.format(ti)
        if key in h_dict:
            h_all.append(h_dict[key])
            groups_all.extend([ti] * len(h_dict[key]))

    H_trained = np.concatenate(h_all, axis=0).astype(np.float64)
    trial_groups = np.array(groups_all)

    # DIAGNOSTICS
    log.info("  H_trained: shape=%s, range=[%.4f, %.4f]",
             H_trained.shape, H_trained.min(), H_trained.max())
    log.info("  NaN count: %d", np.isnan(H_trained).sum())
    zero_var_cols = (H_trained.std(axis=0) < 1e-10).sum()
    log.info("  Zero-variance columns: %d/%d", zero_var_cols, H_trained.shape[1])

    return model, H_trained, trial_groups, cc, n_epochs, passed


# =========================================================================
# STEP 4: Compute biological probe variables (ONCE — same for all tasks)
# =========================================================================

def compute_all_bio_targets(trials):
    """Compute 7 biological variables from input data.

    Uses the EXACT same compute_bio_variables function from v3.
    These are computed from INPUT only — identical across all three tasks.

    Returns: bio_dict {var_name: (N_total,) array}, bio_names list
    """
    log.info("Computing biological probe variables...")

    bio_all = []
    bio_names = None

    for ti, t in enumerate(trials):
        bio, names = compute_bio_variables(t['X'], t['Y'], t['epoch'], t.get('meta', {}))
        if bio_names is None:
            bio_names = names
        bio_all.append(bio)

    bio_concat = np.nan_to_num(
        np.concatenate(bio_all, axis=0).astype(np.float64), nan=0.0)

    bio_dict = {}
    for vi, vname in enumerate(bio_names):
        bio_dict[vname] = bio_concat[:, vi]

    # DIAGNOSTICS
    for vname in bio_names:
        v = bio_dict[vname]
        log.info("  %-25s shape=%s range=[%.4f, %.4f] std=%.4f",
                 vname, v.shape, v.min(), v.max(), v.std())

    return bio_dict, bio_names


# =========================================================================
# STEP 5: Run iAAFT probing for one task (IMPORT, DO NOT REIMPLEMENT)
# =========================================================================

def run_probing_for_task(task_name, H_trained, bio_dict, bio_names,
                         trial_groups, n_surrogates=50):
    """Run unified iAAFT screening + resample ablation for one task.

    Calls the EXACT same iaaft_screen and resample_ablation functions
    from run_synthetic_reality_v3_iaaft.py.

    Returns dict with screening results, ablation results, mandatory list.
    """
    log.info("\n" + "=" * 60)
    log.info("PROBING: %s", task_name)
    log.info("=" * 60)

    screen_results = {}
    ablation_results = {}
    r2_values = {}

    # iAAFT screening
    log.info("  iAAFT screening (%d surrogates, R2 floor=0.01)...", n_surrogates)
    for vi, vname in enumerate(bio_names):
        target = bio_dict[vname]
        result = iaaft_screen(H_trained, target, trial_groups,
                              n_surrogates=n_surrogates, random_state=42 + vi)
        screen_results[vname] = result
        r2_values[vname] = result['r2_trained']

        status = "PASS" if result['passes_screen'] else "FAIL"
        reason = ""
        if not result.get('passes_floor', True):
            reason = " (R2<0.01)"
        elif not result.get('passes_iaaft', True):
            reason = " (p>=0.05)"
        log.info("    %-25s R2=%.4f  95th=%.4f  p=%.3f  [%s%s]",
                 vname, result['r2_trained'], result.get('iaaft_95th', 0),
                 result.get('p_value', 1), status, reason)

    n_pass = sum(1 for r in screen_results.values() if r['passes_screen'])
    log.info("  iAAFT screening: %d/%d passed dual gate", n_pass, len(bio_names))

    # Resample ablation on survivors
    log.info("  Resample ablation on %d survivors...", n_pass)
    abl_rng = np.random.default_rng(42)
    for vname in bio_names:
        if screen_results[vname]['passes_screen']:
            target = bio_dict[vname]
            abl = resample_ablation(H_trained, target, trial_groups, rng=abl_rng)
            ablation_results[vname] = abl
            log.info("    %-25s %s (z: %s)",
                     vname, abl['overall_verdict'],
                     ', '.join('{:.1f}'.format(r['z_score'])
                               for r in abl['per_k'].values()))
        else:
            ablation_results[vname] = {
                'overall_verdict': 'NOT_TESTED',
                'reason': 'Failed iAAFT screening',
            }

    mandatory = [v for v, r in ablation_results.items()
                 if r.get('overall_verdict') == 'MANDATORY']

    log.info("\n  --- %s SUMMARY ---", task_name)
    log.info("  iAAFT passed: %d/7", n_pass)
    log.info("  Mandatory: %s", ', '.join(mandatory) if mandatory else 'NONE')

    return {
        'screening': {v: {k: val for k, val in r.items() if k != 'null_r2s'}
                      for v, r in screen_results.items()},
        'ablation': ablation_results,
        'mandatory': mandatory,
        'r2_trained': r2_values,
        'n_passed_screen': n_pass,
    }


# =========================================================================
# STEP 6: Comparison table and prediction evaluation
# =========================================================================

def print_comparison_table(all_results, bio_names):
    """Print the critical comparison table across all three tasks."""
    log.info("\n" + "=" * 90)
    log.info("TASK-DEPENDENCE v2 COMPARISON TABLE")
    log.info("=" * 90)
    log.info("%-25s | %-16s | %-18s | %-18s",
             'Variable', 'A (29n spikes)', 'B2 (mean rate)', 'C2 (epoch 5-cls)')
    log.info("-" * 90)

    for vname in bio_names:
        row = "%-25s" % vname
        for task_name in ['task_a', 'task_b2', 'task_c2']:
            tr = all_results.get(task_name, {})
            if tr.get('status') == 'FAILED_QUALITY_GATE':
                row += " | %-16s" % 'FAILED'
            else:
                r2 = tr.get('r2_trained', {}).get(vname, 0)
                is_mand = vname in tr.get('mandatory', [])
                marker = '***' if is_mand else ' - '
                row += " | %7.4f %s     " % (r2, marker)
        log.info(row)

    log.info("-" * 90)
    log.info("  *** = MANDATORY   - = not mandatory/failed screen")


def evaluate_predictions(all_results):
    """Evaluate the key predictions for v2 task design."""
    log.info("\n" + "=" * 80)
    log.info("PREDICTION CHECK (v2)")
    log.info("=" * 80)

    task_a_mand = set(all_results.get('task_a', {}).get('mandatory', []))
    task_b2_mand = set(all_results.get('task_b2', {}).get('mandatory', []))
    task_c2_mand = set(all_results.get('task_c2', {}).get('mandatory', []))

    # Prediction 1: firing_rate_input
    # Task A (spikes): MANDATORY (tautological — output IS spike trains)
    # Task B2 (mean rate): may survive (output still contains rate info, just scalar)
    # Task C2 (epoch): should DROP OUT (epoch labels have no rate information)
    fri_a = 'firing_rate_input' in task_a_mand
    fri_b2 = 'firing_rate_input' in task_b2_mand
    fri_c2 = 'firing_rate_input' in task_c2_mand

    log.info("  firing_rate_input:")
    log.info("    Task A  (29n spike trains):  %s", "MANDATORY" if fri_a else "not mandatory")
    log.info("    Task B2 (output mean rate):  %s", "MANDATORY" if fri_b2 else "not mandatory")
    log.info("    Task C2 (epoch identity):    %s", "MANDATORY" if fri_c2 else "not mandatory")

    if fri_a and not fri_c2:
        log.info("  >>> firing_rate_input drops out when output has no rate info")
        if fri_b2:
            log.info("  >>> but survives for scalar rate prediction — rate info still in output")
        log.info("  >>> CONCLUSION: firing_rate_input is OUTPUT-FORMAT-DEPENDENT")
    elif fri_a and fri_c2:
        log.info("  >>> PREDICTION REFUTED: firing_rate_input mandatory even for epoch prediction")
        log.info("  >>> May be genuinely important for computation")

    # Prediction 2: theta_gamma_pac should survive across all tasks
    # It reflects temporal world structure, not output format
    tgp_a = 'theta_gamma_pac' in task_a_mand
    tgp_b2 = 'theta_gamma_pac' in task_b2_mand
    tgp_c2 = 'theta_gamma_pac' in task_c2_mand

    log.info("")
    log.info("  theta_gamma_pac:")
    log.info("    Task A  (29n spike trains):  %s", "MANDATORY" if tgp_a else "not mandatory")
    log.info("    Task B2 (output mean rate):  %s", "MANDATORY" if tgp_b2 else "not mandatory")
    log.info("    Task C2 (epoch identity):    %s", "MANDATORY" if tgp_c2 else "not mandatory")

    if tgp_a and tgp_c2:
        log.info("  >>> PREDICTION CONFIRMED: theta_gamma_pac is REALITY-REFLECTIVE")
        log.info("  >>> Survives task change — reflects temporal world structure")
    elif tgp_a and not tgp_c2:
        log.info("  >>> PREDICTION PARTIALLY REFUTED: theta_gamma_pac is task-dependent")

    # Classify all variables across tasks
    log.info("\n" + "=" * 80)
    log.info("VARIABLE CLASSIFICATION")
    log.info("=" * 80)

    # A variable is "rate-tautological" if mandatory in A but not C2 (epoch)
    # It's "reality-reflective" if mandatory in both A and C2
    rate_tautological = task_a_mand - task_c2_mand
    reality_reflective = task_a_mand & task_c2_mand
    c2_only = task_c2_mand - task_a_mand  # mandatory only for epoch task

    log.info("  RATE-TAUTOLOGICAL (Task A only, not C2):  %s", sorted(rate_tautological))
    log.info("  REALITY-REFLECTIVE (both A and C2):       %s", sorted(reality_reflective))
    log.info("  EPOCH-SPECIFIC (C2 only, not A):          %s", sorted(c2_only))
    log.info("  Task B2 mandatory:                        %s", sorted(task_b2_mand))

    return {
        'rate_tautological': sorted(rate_tautological),
        'reality_reflective': sorted(reality_reflective),
        'epoch_specific': sorted(c2_only),
        'task_a_mandatory': sorted(task_a_mand),
        'task_b2_mandatory': sorted(task_b2_mand),
        'task_c2_mandatory': sorted(task_c2_mand),
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESCARTES Task-Dependence Test')
    parser.add_argument('--processed-dir', required=True)
    parser.add_argument('--output-dir', default='results/task_dependence')
    parser.add_argument('--source-subject', default='5')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-surrogates', type=int, default=50)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("TASK-DEPENDENCE TEST v2")
    log.info("  Question: Is firing_rate_input mandatory because of biology")
    log.info("            or because the output format trivially contains it?")
    log.info("  Method: Same input, three different prediction targets")
    log.info("  Tasks:")
    log.info("    A:  29-neuron spike prediction (control — output contains rates)")
    log.info("    B2: output mean rate (scalar — still contains rate info)")
    log.info("    C2: temporal epoch (5-class — NO rate info in output)")
    log.info("  Key prediction: firing_rate_input drops out of C2 (no rates)")
    log.info("                  theta_gamma_pac survives all tasks")
    log.info("  Subject: sub-%s", args.source_subject)
    log.info("  Device: %s", args.device)
    log.info("  iAAFT surrogates: %d", args.n_surrogates)
    log.info("=" * 70)

    t_start = time.time()

    # Step 1: Load data
    trials, meta = load_sub5_data(args.processed_dir, args.source_subject)

    # Step 2: Prepare targets (v2: learnable per-timestep targets)
    tasks = prepare_task_targets(trials)

    # Step 3: Compute bio probe variables (ONCE — same for all tasks)
    bio_dict, bio_names = compute_all_bio_targets(trials)
    log.info("  Probe variables: %s", bio_names)

    # Step 4: Train and probe each task
    task_configs = [
        ('task_a', 'Spike prediction (29 neurons — output contains firing rates)'),
        ('task_b2', 'Output mean rate (scalar per timestep — rate info, no spike format)'),
        ('task_c2', 'Epoch prediction (5-class — NO rate info, temporal structure only)'),
    ]

    all_results = {}
    for task_name, description in task_configs:
        log.info("\n" + "=" * 70)
        log.info("TASK: %s", description)
        log.info("=" * 70)

        # Train
        model, H_trained, trial_groups, cc, n_epochs, passed = \
            train_task_lstm(trials, tasks[task_name], task_name,
                            hidden_dim=args.hidden_dim, device=args.device)

        if not passed:
            log.warning("  %s FAILED quality gate (CC=%.3f) -- results may be unreliable",
                        task_name, cc)
            # Still run probing — low CC doesn't mean hidden states are empty,
            # just that the model struggled with the task
            log.info("  Proceeding with probing despite low CC...")

        # Ensure H_trained and bio targets are aligned in length
        n_samples = min(len(H_trained), len(bio_dict[bio_names[0]]))
        H_trimmed = H_trained[:n_samples]
        bio_trimmed = {k: v[:n_samples] for k, v in bio_dict.items()}
        groups_trimmed = trial_groups[:n_samples]

        log.info("  Aligned samples: %d (H) x %d (bio)",
                 H_trimmed.shape[0], n_samples)

        # Probe
        results = run_probing_for_task(
            task_name, H_trimmed, bio_trimmed, bio_names,
            groups_trimmed, n_surrogates=args.n_surrogates)

        results['cc'] = cc
        results['n_epochs'] = n_epochs
        results['quality_passed'] = passed
        all_results[task_name] = results

    # Step 5: Comparison table
    print_comparison_table(all_results, bio_names)

    # Step 6: Evaluate predictions
    classification = evaluate_predictions(all_results)

    # Save results
    total_time = time.time() - t_start

    save_data = {
        'experiment': 'task_dependence_test_v2',
        'source_subject': args.source_subject,
        'hidden_dim': args.hidden_dim,
        'n_surrogates': args.n_surrogates,
        'total_time_min': total_time / 60,
        'task_a_mandatory': all_results.get('task_a', {}).get('mandatory', []),
        'task_b2_mandatory': all_results.get('task_b2', {}).get('mandatory', []),
        'task_c2_mandatory': all_results.get('task_c2', {}).get('mandatory', []),
        'classification': classification,
        'task_details': {
            task_name: {
                'cc': r.get('cc', 0),
                'n_epochs': r.get('n_epochs', 0),
                'quality_passed': r.get('quality_passed', False),
                'mandatory': r.get('mandatory', []),
                'n_passed_screen': r.get('n_passed_screen', 0),
                'r2_trained': r.get('r2_trained', {}),
            }
            for task_name, r in all_results.items()
        },
    }

    save_path = output_dir / 'task_dependence_results.json'
    with open(save_path, 'w') as f:
        json.dump(_convert_numpy(save_data), f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("TASK-DEPENDENCE TEST COMPLETE (%.1f min)", total_time / 60)
    log.info("Results: %s", save_path)
    log.info("=" * 70)


if __name__ == '__main__':
    main()
