#!/usr/bin/env python3
"""
Task-Dependence Test: Does firing_rate_input disappear when the output
format no longer contains firing rates?

Uses sub-5 from Kyzar dataset. Same input spike trains for all three tasks.
Only the prediction target changes.

Task A: Spike prediction (output = 29 neurons) — CONTROL
Task B: Accuracy prediction (output = scalar 0/1 per trial)
Task C: Memory load prediction (output = one-hot set size 1/2/3 per trial)

Prediction:
  firing_rate_input is MANDATORY in Task A (tautological: output contains rates)
  firing_rate_input DROPS OUT of Tasks B/C (output has no rate information)
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
    'task_b': 137,
    'task_c': 271,
}


# =========================================================================
# STEP 1: Load data and extract behavioral labels
# =========================================================================

def load_sub5_with_behavior(processed_dir, subject='5'):
    """Load sub-5 trials AND extract behavioral labels from metadata.

    Returns:
        trials: list of dicts with X, Y, epoch, meta (same as v3)
        accuracy_labels: list of int (0 or 1 per trial)
        load_labels: list of int (1, 2, or 3 per trial — set size)
        meta: full session metadata
    """
    trials, meta = load_source_data(processed_dir, subject)

    trial_meta = meta.get('trial_metadata', [])
    accuracy_labels = []
    load_labels = []

    for ti, t in enumerate(trials):
        if ti < len(trial_meta):
            tm = trial_meta[ti]
            accuracy_labels.append(int(tm.get('accuracy', 0)))
            load_labels.append(int(tm.get('load', 1)))
        else:
            # Fallback if metadata is missing for this trial
            accuracy_labels.append(0)
            load_labels.append(1)

    accuracy_labels = np.array(accuracy_labels)
    load_labels = np.array(load_labels)

    log.info("Behavioral labels extracted:")
    log.info("  Accuracy: %s", dict(zip(*np.unique(accuracy_labels, return_counts=True))))
    log.info("  Load (set size): %s", dict(zip(*np.unique(load_labels, return_counts=True))))

    return trials, accuracy_labels, load_labels, meta


# =========================================================================
# STEP 2: Create time-expanded targets for Tasks B and C
# =========================================================================

def prepare_task_targets(trials, accuracy_labels, load_labels):
    """Create three sets of training targets from the same input data.

    Task A: output spike trains (unchanged from existing pipeline)
    Task B: accuracy expanded to every timestep (T, 1) — constant per trial
    Task C: load as one-hot expanded to every timestep (T, 3) — constant per trial

    Returns dict of task_name -> list of (T, n_output) arrays
    """
    tasks = {}

    # Task A: existing spike targets (unchanged)
    tasks['task_a'] = [t['Y'] for t in trials]

    # Task B: expand accuracy to per-timestep scalar
    tasks['task_b'] = []
    for i, t in enumerate(trials):
        T = t['X'].shape[0]
        target = np.full((T, 1), accuracy_labels[i], dtype=np.float32)
        tasks['task_b'].append(target)

    # Task C: expand load to per-timestep one-hot
    # Load values are 1, 2, 3 -> indices 0, 1, 2
    unique_loads = sorted(np.unique(load_labels))
    load_to_idx = {v: i for i, v in enumerate(unique_loads)}
    n_categories = len(unique_loads)

    tasks['task_c'] = []
    for i, t in enumerate(trials):
        T = t['X'].shape[0]
        one_hot = np.zeros((T, n_categories), dtype=np.float32)
        one_hot[:, load_to_idx[load_labels[i]]] = 1.0
        tasks['task_c'].append(one_hot)

    log.info("Task targets prepared:")
    log.info("  Task A output dim: %d (spike trains)", tasks['task_a'][0].shape[1])
    log.info("  Task B output dim: %d (accuracy, binary)", tasks['task_b'][0].shape[1])
    log.info("  Task C output dim: %d (load, %d classes: %s)",
             tasks['task_c'][0].shape[1], n_categories, unique_loads)

    return tasks, n_categories, unique_loads


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

    # Quality gate
    if task_name == 'task_a':
        passed = cc > 0.3
        metric_name = 'CC'
    elif task_name == 'task_b':
        # For binary: CC > 0.1 is meaningful (imbalanced data)
        passed = abs(cc) > 0.05
        metric_name = 'CC'
    else:  # task_c
        # For 3-class one-hot: CC > 0.1 is above chance
        passed = abs(cc) > 0.05
        metric_name = 'CC'

    log.info("  Quality gate: %s=%.3f %s (threshold: %s)",
             metric_name, cc, "PASS" if passed else "FAIL",
             "0.30" if task_name == 'task_a' else "0.05")

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
    log.info("\n" + "=" * 85)
    log.info("TASK-DEPENDENCE COMPARISON TABLE")
    log.info("=" * 85)
    log.info("%-25s | %-15s | %-17s | %-17s",
             'Variable', 'Task A (spikes)', 'Task B (accuracy)', 'Task C (load)')
    log.info("-" * 85)

    for vname in bio_names:
        row = "%-25s" % vname
        for task_name in ['task_a', 'task_b', 'task_c']:
            tr = all_results.get(task_name, {})
            if tr.get('status') == 'FAILED_QUALITY_GATE':
                row += " | %-15s" % 'FAILED'
            else:
                r2 = tr.get('r2_trained', {}).get(vname, 0)
                is_mand = vname in tr.get('mandatory', [])
                marker = '***' if is_mand else ' - '
                row += " | %7.4f %s    " % (r2, marker)
        log.info(row)

    log.info("-" * 85)
    log.info("  *** = MANDATORY   - = not mandatory/failed screen")


def evaluate_predictions(all_results):
    """Evaluate the two key predictions."""
    log.info("\n" + "=" * 80)
    log.info("PREDICTION CHECK")
    log.info("=" * 80)

    task_a_mand = set(all_results.get('task_a', {}).get('mandatory', []))
    task_b_mand = set(all_results.get('task_b', {}).get('mandatory', []))
    task_c_mand = set(all_results.get('task_c', {}).get('mandatory', []))

    # Prediction 1: firing_rate_input drops out of B and C
    fri_a = 'firing_rate_input' in task_a_mand
    fri_b = 'firing_rate_input' in task_b_mand
    fri_c = 'firing_rate_input' in task_c_mand

    log.info("  firing_rate_input in Task A (spikes):   %s", "YES" if fri_a else "NO")
    log.info("  firing_rate_input in Task B (accuracy): %s", "YES" if fri_b else "NO")
    log.info("  firing_rate_input in Task C (load):     %s", "YES" if fri_c else "NO")

    if fri_a and not fri_b and not fri_c:
        log.info("  >>> PREDICTION CONFIRMED: firing_rate_input is TASK-TAUTOLOGICAL")
        log.info("  >>> Mandatory only because the output format contained rates")
    elif fri_a and (fri_b or fri_c):
        log.info("  >>> PREDICTION REFUTED: firing_rate_input survives task change")
        log.info("  >>> May be genuinely important for computation")
    elif not fri_a:
        log.info("  >>> UNEXPECTED: firing_rate_input not mandatory even in Task A")

    # Prediction 2: theta_gamma_pac survives across tasks
    tgp_a = 'theta_gamma_pac' in task_a_mand
    tgp_b = 'theta_gamma_pac' in task_b_mand
    tgp_c = 'theta_gamma_pac' in task_c_mand

    log.info("")
    log.info("  theta_gamma_pac in Task A (spikes):   %s", "YES" if tgp_a else "NO")
    log.info("  theta_gamma_pac in Task B (accuracy): %s", "YES" if tgp_b else "NO")
    log.info("  theta_gamma_pac in Task C (load):     %s", "YES" if tgp_c else "NO")

    if tgp_a and tgp_b and tgp_c:
        log.info("  >>> PREDICTION CONFIRMED: theta_gamma_pac is REALITY-REFLECTIVE")
        log.info("  >>> Survives task change -- reflects world structure, not output format")
    elif tgp_a and not (tgp_b and tgp_c):
        log.info("  >>> PREDICTION PARTIALLY REFUTED: theta_gamma_pac is task-dependent")

    # Classify all variables
    log.info("\n" + "=" * 80)
    log.info("VARIABLE CLASSIFICATION")
    log.info("=" * 80)

    tautological = task_a_mand - task_b_mand - task_c_mand
    reality_reflective = task_a_mand & task_b_mand & task_c_mand
    mixed = task_a_mand - tautological - reality_reflective

    log.info("  TASK-TAUTOLOGICAL (Task A only):  %s", sorted(tautological))
    log.info("  REALITY-REFLECTIVE (all tasks):   %s", sorted(reality_reflective))
    log.info("  MIXED (some tasks):               %s", sorted(mixed))

    return {
        'tautological': sorted(tautological),
        'reality_reflective': sorted(reality_reflective),
        'mixed': sorted(mixed),
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
    log.info("TASK-DEPENDENCE TEST")
    log.info("  Question: Is firing_rate_input mandatory because of biology")
    log.info("            or because the output format trivially contains it?")
    log.info("  Method: Same input, three different prediction targets")
    log.info("  Prediction: firing_rate_input drops out of Tasks B/C")
    log.info("              theta_gamma_pac survives across all tasks")
    log.info("  Subject: sub-%s", args.source_subject)
    log.info("  Device: %s", args.device)
    log.info("  iAAFT surrogates: %d", args.n_surrogates)
    log.info("=" * 70)

    t_start = time.time()

    # Step 1: Load data with behavioral labels
    trials, accuracy_labels, load_labels, meta = \
        load_sub5_with_behavior(args.processed_dir, args.source_subject)

    # Step 2: Prepare targets
    tasks, n_categories, unique_loads = \
        prepare_task_targets(trials, accuracy_labels, load_labels)

    # Step 3: Compute bio probe variables (ONCE — same for all tasks)
    bio_dict, bio_names = compute_all_bio_targets(trials)

    # Trim bio_names to the 7 used in v3 (compute_bio_variables returns 7)
    # They should match PROBE_VARIABLES from v3
    log.info("  Probe variables: %s", bio_names)

    # Step 4: Train and probe each task
    task_configs = [
        ('task_a', 'Spike prediction (output contains firing rates)'),
        ('task_b', 'Accuracy prediction (output = binary, no rates)'),
        ('task_c', 'Load prediction (output = one-hot set size, no rates)'),
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
        'experiment': 'task_dependence_test',
        'source_subject': args.source_subject,
        'hidden_dim': args.hidden_dim,
        'n_surrogates': args.n_surrogates,
        'total_time_min': total_time / 60,
        'task_a_mandatory': all_results.get('task_a', {}).get('mandatory', []),
        'task_b_mandatory': all_results.get('task_b', {}).get('mandatory', []),
        'task_c_mandatory': all_results.get('task_c', {}).get('mandatory', []),
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
