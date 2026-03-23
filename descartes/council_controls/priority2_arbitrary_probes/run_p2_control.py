"""
run_p2_control.py

Priority 2: Arbitrary Probe Specificity Control

Tests whether probing results are specific to biophysical variables or
whether arbitrary targets produce equivalent results.

Protocol:
  1. Generate 3 types of arbitrary targets:
     - Random linear projections of hidden states
     - PCA components of the input signal
     - Unrelated-domain variables (Lorenz, random walk, sinusoidal)
  2. Run probing on synthetic reality v2 conditions with arbitrary targets
     using the same Ridge delta-R2 + GroupKFold + iAAFT pipeline
  3. Decision: specificity confirmed if arbitrary probes show DIFFERENT
     mandatory target sets per condition than the biophysical probes

Specificity criteria:
  - Biophysical mandatory set != arbitrary mandatory set per condition
  - Random projections should decode well (they are linear functions of H)
    but should NOT show the same condition-dependent patterns
  - Unrelated-domain targets should decode poorly (low delta-R2)
  - If unrelated targets achieve comparable delta-R2 to biophysical ones,
    probing specificity is FAILED

Saves results to results/council_controls/p2_arbitrary_control.json

No external dependencies beyond numpy, scipy, sklearn.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from descartes.council_controls.priority2_arbitrary_probes.arbitrary_targets import (
    generate_all_arbitrary_targets,
    get_target_type,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5
DELTA_THRESHOLD = 0.1
IAAFT_ITERATIONS = 20
IAAFT_N_SURROGATES = 50  # Reduced for speed in control runs
SPECIFICITY_MARGIN = 0.05  # Delta-R2 margin for specificity comparison


# ---------------------------------------------------------------------------
# iAAFT surrogate (local copy to avoid circular imports)
# ---------------------------------------------------------------------------

def _iaaft_surrogate(signal, n_iterations=IAAFT_ITERATIONS, rng=None):
    """Generate iAAFT surrogate preserving spectrum + amplitude distribution."""
    if rng is None:
        rng = np.random.RandomState(42)
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


# ---------------------------------------------------------------------------
# Single-target probing with GroupKFold + iAAFT
# ---------------------------------------------------------------------------

def probe_single_target(
    H_trained: np.ndarray,
    H_untrained: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    target_name: str,
) -> dict:
    """Probe one target variable with Ridge delta-R2 + GroupKFold + iAAFT.

    Parameters
    ----------
    H_trained : ndarray, (n_samples, n_hidden)
    H_untrained : ndarray, (n_samples, n_hidden)
    target : ndarray, (n_samples,)
    groups : ndarray, (n_samples,)
    target_name : str

    Returns
    -------
    result : dict with delta_R2, iaaft_p, classification
    """
    n_groups = len(np.unique(groups))
    actual_splits = min(CV_FOLDS, n_groups)
    if actual_splits < 2:
        return {
            'target_name': target_name,
            'target_type': get_target_type(target_name),
            'delta_R2': 0.0, 'R2_trained': 0.0, 'R2_untrained': 0.0,
            'iaaft_p': 1.0, 'classification': 'INSUFFICIENT_DATA',
        }

    gkf = GroupKFold(n_splits=actual_splits)

    def _cv_r2(H, y):
        fold_r2s = []
        for train_idx, test_idx in gkf.split(H, y, groups):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(H[train_idx])
            X_test = scaler.transform(H[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]
            y_std = y_train.std()
            if y_std < 1e-10:
                fold_r2s.append(0.0)
                continue
            y_train_s = (y_train - y_train.mean()) / y_std
            y_test_s = (y_test - y_train.mean()) / y_std
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit(X_train, y_train_s)
            fold_r2s.append(float(ridge.score(X_test, y_test_s)))
        return float(np.mean(fold_r2s)) if fold_r2s else 0.0

    r2_trained = _cv_r2(H_trained, target)
    r2_untrained = _cv_r2(H_untrained, target)
    delta_r2 = r2_trained - r2_untrained

    # iAAFT null distribution
    rng = np.random.RandomState(hash(target_name) % (2**31))
    null_r2s = []
    for i in range(IAAFT_N_SURROGATES):
        surr = _iaaft_surrogate(
            target, rng=np.random.RandomState(rng.randint(0, 2**31))
        )
        null_r2s.append(_cv_r2(H_trained, surr))

    null_r2s = np.array(null_r2s)
    iaaft_p = float((np.sum(null_r2s >= r2_trained) + 1) /
                    (IAAFT_N_SURROGATES + 1))

    # Classification
    if delta_r2 > DELTA_THRESHOLD and iaaft_p < 0.05:
        classification = 'DECODABLE'
    elif delta_r2 > DELTA_THRESHOLD:
        classification = 'DECODABLE_NOT_TEMPORAL'
    else:
        classification = 'NOT_DECODABLE'

    return {
        'target_name': target_name,
        'target_type': get_target_type(target_name),
        'R2_trained': float(r2_trained),
        'R2_untrained': float(r2_untrained),
        'delta_R2': float(delta_r2),
        'iaaft_p': float(iaaft_p),
        'iaaft_null_mean': float(np.mean(null_r2s)),
        'iaaft_null_std': float(np.std(null_r2s)),
        'classification': classification,
    }


# ---------------------------------------------------------------------------
# Per-condition probing
# ---------------------------------------------------------------------------

def probe_condition(
    condition_name: str,
    H_trained: np.ndarray,
    H_untrained: np.ndarray,
    X_input: np.ndarray,
    biophysical_targets: Dict[str, np.ndarray],
    groups: np.ndarray,
) -> dict:
    """Probe one condition with both biophysical and arbitrary targets.

    Parameters
    ----------
    condition_name : str
    H_trained, H_untrained : ndarray (n_samples, n_hidden)
    X_input : ndarray (n_samples, n_input)
    biophysical_targets : dict of var_name -> ndarray (n_samples,)
    groups : ndarray (n_samples,)

    Returns
    -------
    condition_result : dict with biophysical and arbitrary probing results.
    """
    logger.info("--- Condition: %s ---", condition_name)
    n_samples = H_trained.shape[0]

    # Generate arbitrary targets for this condition
    arbitrary_targets = generate_all_arbitrary_targets(
        H_trained, X_input, n_samples=n_samples,
    )

    # Probe biophysical targets
    bio_results = {}
    for var_name, target_y in biophysical_targets.items():
        n = min(n_samples, len(target_y))
        if n < 20:
            continue
        result = probe_single_target(
            H_trained[:n], H_untrained[:n], target_y[:n],
            groups[:n], var_name,
        )
        bio_results[var_name] = result
        logger.info("  [bio] %s: dR2=%.3f  iAAFT_p=%.3f  %s",
                     var_name, result['delta_R2'], result['iaaft_p'],
                     result['classification'])

    # Probe arbitrary targets
    arb_results = {}
    for var_name, target_y in arbitrary_targets.items():
        n = min(n_samples, len(target_y))
        if n < 20:
            continue
        result = probe_single_target(
            H_trained[:n], H_untrained[:n], target_y[:n],
            groups[:n], var_name,
        )
        arb_results[var_name] = result
        logger.info("  [arb] %s: dR2=%.3f  iAAFT_p=%.3f  %s",
                     var_name, result['delta_R2'], result['iaaft_p'],
                     result['classification'])

    # Mandatory sets
    bio_mandatory = [
        v for v, r in bio_results.items() if r['classification'] == 'DECODABLE'
    ]
    arb_mandatory = [
        v for v, r in arb_results.items() if r['classification'] == 'DECODABLE'
    ]

    # Aggregate statistics by target type
    type_stats = {}
    for var_name, result in arb_results.items():
        ttype = result['target_type']
        if ttype not in type_stats:
            type_stats[ttype] = {'delta_r2s': [], 'n_decodable': 0, 'total': 0}
        type_stats[ttype]['delta_r2s'].append(result['delta_R2'])
        type_stats[ttype]['total'] += 1
        if result['classification'] == 'DECODABLE':
            type_stats[ttype]['n_decodable'] += 1

    for ttype in type_stats:
        dr2s = type_stats[ttype]['delta_r2s']
        type_stats[ttype]['mean_delta_r2'] = float(np.mean(dr2s))
        type_stats[ttype]['max_delta_r2'] = float(np.max(dr2s))
        type_stats[ttype]['fraction_decodable'] = (
            type_stats[ttype]['n_decodable'] / max(type_stats[ttype]['total'], 1)
        )
        del type_stats[ttype]['delta_r2s']  # Don't serialize the full list

    return {
        'condition': condition_name,
        'n_biophysical': len(bio_results),
        'n_arbitrary': len(arb_results),
        'bio_mandatory': bio_mandatory,
        'arb_mandatory': arb_mandatory,
        'bio_results': bio_results,
        'arb_results': arb_results,
        'type_stats': type_stats,
    }


# ---------------------------------------------------------------------------
# Specificity decision logic
# ---------------------------------------------------------------------------

def assess_specificity(condition_results: List[dict]) -> dict:
    """Assess probe specificity across conditions.

    Decision criteria:
    1. Different mandatory sets: bio_mandatory != arb_mandatory per condition
    2. Unrelated targets should have low delta-R2 (< threshold)
    3. Random projections may decode well but should not be condition-specific
    4. If arbitrary probes reproduce the same condition-dependent pattern
       as biophysical probes, specificity FAILS

    Parameters
    ----------
    condition_results : list of dicts from probe_condition()

    Returns
    -------
    assessment : dict with decision, reasoning, per-condition details.
    """
    n_conditions = len(condition_results)
    if n_conditions == 0:
        return {
            'decision': 'INCONCLUSIVE',
            'reasoning': 'No conditions tested.',
        }

    # Track per-condition mandatory set differences
    condition_specificity = []
    all_bio_mandatory = set()
    unrelated_decodable_counts = []
    random_proj_condition_patterns = []

    for cr in condition_results:
        bio_set = set(cr['bio_mandatory'])
        arb_set = set(cr['arb_mandatory'])
        all_bio_mandatory.update(bio_set)

        # Are the mandatory sets different?
        sets_differ = bio_set != arb_set
        # Specifically: are there bio variables that are NOT in arb set?
        bio_unique = bio_set - arb_set

        # Count unrelated-domain decodable targets
        n_unrelated_decodable = 0
        for vn, vr in cr['arb_results'].items():
            if get_target_type(vn) in ('unrelated_chaotic', 'unrelated_random_walk',
                                        'unrelated_sinusoidal'):
                if vr['classification'] == 'DECODABLE':
                    n_unrelated_decodable += 1
        unrelated_decodable_counts.append(n_unrelated_decodable)

        # Track which random projections are mandatory (condition-specific?)
        rp_mandatory = [
            vn for vn, vr in cr['arb_results'].items()
            if get_target_type(vn) == 'random_projection'
            and vr['classification'] == 'DECODABLE'
        ]
        random_proj_condition_patterns.append(set(rp_mandatory))

        condition_specificity.append({
            'condition': cr['condition'],
            'bio_mandatory': list(bio_set),
            'arb_mandatory': list(arb_set),
            'bio_unique': list(bio_unique),
            'sets_differ': sets_differ,
            'n_unrelated_decodable': n_unrelated_decodable,
        })

    # Aggregate decision
    all_sets_differ = all(cs['sets_differ'] for cs in condition_specificity)
    max_unrelated_decodable = max(unrelated_decodable_counts)
    total_bio_mandatory = len(all_bio_mandatory)

    # Check if random projections show same pattern across conditions
    rp_patterns_identical = False
    if n_conditions >= 2:
        rp_patterns_identical = all(
            p == random_proj_condition_patterns[0]
            for p in random_proj_condition_patterns[1:]
        )

    # Decision
    if all_sets_differ and max_unrelated_decodable == 0 and total_bio_mandatory > 0:
        decision = 'PASS'
        reasoning = (
            f"Specificity confirmed: biophysical mandatory sets differ from "
            f"arbitrary mandatory sets in all {n_conditions} condition(s). "
            f"No unrelated-domain targets are decodable. "
            f"{total_bio_mandatory} biophysical variable(s) are specifically "
            f"encoded."
        )
    elif max_unrelated_decodable > 0:
        decision = 'FAIL'
        reasoning = (
            f"Specificity FAILED: {max_unrelated_decodable} unrelated-domain "
            f"target(s) are decodable from hidden states. This suggests the "
            f"probing methodology does not distinguish biophysical encoding "
            f"from spurious correlations."
        )
    elif not all_sets_differ:
        decision = 'FAIL'
        reasoning = (
            f"Specificity FAILED: biophysical and arbitrary mandatory sets "
            f"are identical in at least one condition. The probing does not "
            f"distinguish biophysical targets from arbitrary ones."
        )
    elif total_bio_mandatory == 0:
        decision = 'INCONCLUSIVE'
        reasoning = (
            f"No biophysical variables were classified as mandatory. "
            f"Cannot assess specificity without positive biophysical results."
        )
    else:
        decision = 'INCONCLUSIVE'
        reasoning = (
            f"Mixed results across {n_conditions} conditions. "
            f"Some conditions show specificity, others do not."
        )

    return {
        'decision': decision,
        'reasoning': reasoning,
        'n_conditions': n_conditions,
        'total_bio_mandatory': total_bio_mandatory,
        'max_unrelated_decodable': max_unrelated_decodable,
        'random_proj_patterns_identical': rp_patterns_identical,
        'per_condition': condition_specificity,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_p2_control(
    conditions: Dict[str, dict],
    biophysical_targets: Dict[str, np.ndarray],
    results_dir: str = 'results/council_controls',
) -> dict:
    """Run full P2 arbitrary probe specificity control.

    Parameters
    ----------
    conditions : dict mapping condition_name -> {
        'H_trained': ndarray (n_samples, n_hidden),
        'H_untrained': ndarray (n_samples, n_hidden),
        'X_input': ndarray (n_samples, n_input),
        'groups': ndarray (n_samples,),
    }
    biophysical_targets : dict mapping var_name -> ndarray (n_samples,)
        Biophysical target variables for comparison.
    results_dir : str

    Returns
    -------
    control_result : dict with specificity assessment and per-condition details.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    condition_results = []

    for cond_name, cond_data in conditions.items():
        H_tr = cond_data['H_trained']
        H_un = cond_data['H_untrained']
        X_in = cond_data['X_input']
        groups = cond_data['groups']

        result = probe_condition(
            cond_name, H_tr, H_un, X_in, biophysical_targets, groups,
        )
        condition_results.append(result)

    # Assess specificity
    assessment = assess_specificity(condition_results)

    # Assemble output
    output = {
        'control': 'P2_ARBITRARY_PROBES',
        'decision': assessment['decision'],
        'reasoning': assessment['reasoning'],
        'specificity_assessment': assessment,
        'per_condition': condition_results,
    }

    # Save
    save_path = results_dir / 'p2_arbitrary_control.json'

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return list(obj)
            return super().default(obj)

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    logger.info("=" * 60)
    logger.info("P2 ARBITRARY PROBE CONTROL: %s", output['decision'])
    logger.info("  Reasoning: %s", output['reasoning'])
    logger.info("Saved to %s", save_path)
    logger.info("=" * 60)

    return output
