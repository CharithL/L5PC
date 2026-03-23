"""
run_p1_control.py

Priority 1: Full Non-Oscillatory Architecture Control

Runs the complete architecture control on circuits C2 and C6:
  - For each circuit x architecture (MLP-50/100/200/500ms, PySR):
    1. Train to equivalent output accuracy
    2. Extract hidden states
    3. Run probing (Ridge delta-R2 with GroupKFold)
    4. Run ablation (resample ablation)
    5. Run iAAFT temporal null
  - Output comparison table: Variable x Architecture
  - Decision logic:
    * architecture_invariant set = mandatory in >= 2 non-LSTM architectures
    * lstm_specific set = mandatory ONLY in LSTM
    * PASS if |architecture_invariant| > 0
    * FAIL if ALL mandatory variables are LSTM-specific

Saves results to results/council_controls/p1_architecture_control.json

No external dependencies beyond torch, sklearn, scipy, numpy.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLP_WINDOW_SIZES_MS = [50, 100, 200, 500]
ARCHITECTURE_NAMES = [f"MLP_{w}ms" for w in MLP_WINDOW_SIZES_MS] + ["PySR"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 5
DELTA_THRESHOLD = 0.1
CAUSAL_Z_THRESHOLD = -2.0
ABLATION_K_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
ABLATION_N_RANDOM = 10
IAAFT_N_SURROGATES = 200
IAAFT_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Ridge delta-R2 probing with GroupKFold
# ---------------------------------------------------------------------------

def ridge_delta_r2_groupkfold(
    H_trained: np.ndarray,
    H_untrained: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alphas: Optional[list] = None,
    n_splits: int = CV_FOLDS,
) -> dict:
    """Ridge delta-R2 probing with GroupKFold cross-validation.

    GroupKFold ensures entire trials stay together, preventing
    temporal autocorrelation leakage between train and test sets.

    Parameters
    ----------
    H_trained : ndarray, (n_samples, n_hidden)
    H_untrained : ndarray, (n_samples, n_hidden)
    target : ndarray, (n_samples,)
    groups : ndarray, (n_samples,)
        Trial/group labels for GroupKFold.
    alphas : list of float
    n_splits : int

    Returns
    -------
    result : dict with R2_trained, R2_untrained, delta_R2, fold details.
    """
    if alphas is None:
        alphas = RIDGE_ALPHAS

    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return {
            'R2_trained': 0.0, 'R2_untrained': 0.0, 'delta_R2': 0.0,
            'n_groups': n_groups, 'error': 'insufficient_groups',
        }

    gkf = GroupKFold(n_splits=actual_splits)

    r2_trained_folds = []
    r2_untrained_folds = []

    for train_idx, test_idx in gkf.split(H_trained, target, groups):
        # Trained hidden states
        scaler_tr = StandardScaler()
        X_tr_train = scaler_tr.fit_transform(H_trained[train_idx])
        X_tr_test = scaler_tr.transform(H_trained[test_idx])

        y_train = target[train_idx]
        y_test = target[test_idx]

        # Standardize target per fold
        y_mean, y_std = y_train.mean(), y_train.std()
        if y_std < 1e-10:
            r2_trained_folds.append(0.0)
            r2_untrained_folds.append(0.0)
            continue

        y_train_s = (y_train - y_mean) / y_std
        y_test_s = (y_test - y_mean) / y_std

        ridge_tr = RidgeCV(alphas=alphas)
        ridge_tr.fit(X_tr_train, y_train_s)
        r2_trained_folds.append(float(ridge_tr.score(X_tr_test, y_test_s)))

        # Untrained hidden states
        scaler_un = StandardScaler()
        X_un_train = scaler_un.fit_transform(H_untrained[train_idx])
        X_un_test = scaler_un.transform(H_untrained[test_idx])

        ridge_un = RidgeCV(alphas=alphas)
        ridge_un.fit(X_un_train, y_train_s)
        r2_untrained_folds.append(float(ridge_un.score(X_un_test, y_test_s)))

    r2_tr = float(np.mean(r2_trained_folds))
    r2_un = float(np.mean(r2_untrained_folds))

    return {
        'R2_trained': r2_tr,
        'R2_untrained': r2_un,
        'delta_R2': r2_tr - r2_un,
        'fold_R2_trained': r2_trained_folds,
        'fold_R2_untrained': r2_untrained_folds,
        'n_groups': n_groups,
    }


# ---------------------------------------------------------------------------
# iAAFT null distribution
# ---------------------------------------------------------------------------

def iaaft_surrogate(signal: np.ndarray, n_iterations: int = IAAFT_ITERATIONS,
                    rng: np.random.RandomState = None) -> np.ndarray:
    """Generate an iAAFT surrogate preserving power spectrum + amplitude dist.

    Parameters
    ----------
    signal : ndarray, (T,)
    n_iterations : int
    rng : RandomState

    Returns
    -------
    surrogate : ndarray, (T,)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(signal)
    fft_orig = np.fft.rfft(signal)
    amplitudes = np.abs(fft_orig)
    sorted_original = np.sort(signal)

    # Initial: random phase
    random_phases = rng.uniform(0, 2 * np.pi, size=len(fft_orig))
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0

    surrogate = np.fft.irfft(amplitudes * np.exp(1j * random_phases), n=n)

    for _ in range(n_iterations):
        # Match amplitude distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_original[rank_order]
        # Match power spectrum
        fft_surr = np.fft.rfft(surrogate)
        phases_surr = np.angle(fft_surr)
        surrogate = np.fft.irfft(amplitudes * np.exp(1j * phases_surr), n=n)

    return surrogate


def iaaft_null_distribution(
    H: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    n_surrogates: int = IAAFT_N_SURROGATES,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, float]:
    """Build iAAFT null distribution for delta-R2.

    Generates phase-randomized surrogates of the target variable,
    preserving its power spectrum and amplitude distribution. Probes
    each surrogate and builds a null distribution of R2 values.

    Parameters
    ----------
    H : ndarray, (n_samples, n_hidden)
        Trained hidden states.
    target : ndarray, (n_samples,)
    groups : ndarray, (n_samples,)
    n_surrogates : int
    random_state : int

    Returns
    -------
    observed_r2 : float
    null_r2s : ndarray, (n_surrogates,)
    p_value : float
    """
    rng = np.random.RandomState(random_state)

    # Observed R2 (just trained, since we compare against shuffled targets)
    n_groups = len(np.unique(groups))
    actual_splits = min(CV_FOLDS, n_groups)
    if actual_splits < 2:
        return 0.0, np.zeros(n_surrogates), 1.0

    gkf = GroupKFold(n_splits=actual_splits)

    def _compute_r2(y):
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
        return float(np.mean(fold_r2s))

    observed_r2 = _compute_r2(target)

    null_r2s = np.zeros(n_surrogates)
    for i in range(n_surrogates):
        surrogate_target = iaaft_surrogate(
            target, n_iterations=IAAFT_ITERATIONS,
            rng=np.random.RandomState(rng.randint(0, 2**31))
        )
        null_r2s[i] = _compute_r2(surrogate_target)

    p_value = float((np.sum(null_r2s >= observed_r2) + 1) / (n_surrogates + 1))
    return observed_r2, null_r2s, p_value


# ---------------------------------------------------------------------------
# Resample ablation (architecture-agnostic version)
# ---------------------------------------------------------------------------

def resample_ablation_generic(
    H: np.ndarray,
    target: np.ndarray,
    output_pred: np.ndarray,
    groups: np.ndarray,
    k_fractions: Optional[list] = None,
    n_random_repeats: int = ABLATION_N_RANDOM,
    random_state: int = 42,
) -> Tuple[list, float]:
    """Architecture-agnostic resample ablation using probing degradation.

    Since MLP and PySR do not have recurrent hidden states that can be
    clamped during a forward pass, we use a probe-based approach:
    - Train a Ridge probe from H -> output
    - For each k%, replace the top-k% target-correlated hidden dims
      in H with random empirical samples, re-predict output from the
      ablated H, measure degradation.
    - Compare against random dim ablation.

    Parameters
    ----------
    H : ndarray, (n_samples, n_hidden)
    target : ndarray, (n_samples,)
        Biophysical variable for correlation-ranked dim selection.
    output_pred : ndarray, (n_samples,) or (n_samples, n_out)
        Model output predictions for measuring degradation.
    groups : ndarray, (n_samples,)
    k_fractions : list of float
    n_random_repeats : int
    random_state : int

    Returns
    -------
    results : list of dict per k value
    baseline_r2 : float (probe R2 with intact H)
    """
    if k_fractions is None:
        k_fractions = ABLATION_K_FRACTIONS

    rng = np.random.RandomState(random_state)
    n_hidden = H.shape[1]

    if output_pred.ndim > 1:
        output_target = output_pred[:, 0]
    else:
        output_target = output_pred

    # Correlation of each hidden dim with the biophysical target
    dim_corrs = np.array([
        abs(float(np.corrcoef(H[:, d], target)[0, 1]))
        if np.std(H[:, d]) > 1e-10 else 0.0
        for d in range(n_hidden)
    ])
    sorted_dims = np.argsort(dim_corrs)[::-1]

    # Baseline: probe R2 from intact H -> output
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(H_scaled, output_target)
    baseline_r2 = float(ridge.score(H_scaled, output_target))

    results = []
    for k_frac in k_fractions:
        n_ablate = max(1, int(round(k_frac * n_hidden)))

        # Target-correlated ablation
        target_dims = sorted_dims[:n_ablate]
        H_ablated = H_scaled.copy()
        for d in target_dims:
            H_ablated[:, d] = rng.choice(H_scaled[:, d], size=len(H_scaled),
                                          replace=True)
        target_r2 = float(ridge.score(H_ablated, output_target))

        # Random ablation
        random_r2s = []
        for _ in range(n_random_repeats):
            rand_dims = rng.choice(n_hidden, size=n_ablate, replace=False)
            H_rand = H_scaled.copy()
            for d in rand_dims:
                H_rand[:, d] = rng.choice(H_scaled[:, d], size=len(H_scaled),
                                           replace=True)
            random_r2s.append(float(ridge.score(H_rand, output_target)))

        random_mean = float(np.mean(random_r2s))
        random_std = float(np.std(random_r2s))

        if random_std > 1e-10:
            z_score = (target_r2 - random_mean) / random_std
        else:
            z_score = -10.0 if target_r2 < random_mean else 0.0

        verdict = 'CAUSAL' if z_score < CAUSAL_Z_THRESHOLD else 'NON_CAUSAL'

        results.append({
            'k_frac': float(k_frac),
            'n_ablated': int(n_ablate),
            'target_r2': float(target_r2),
            'target_r2_drop': float(baseline_r2 - target_r2),
            'random_r2_mean': random_mean,
            'random_r2_std': random_std,
            'z_score': float(z_score),
            'verdict': verdict,
        })

    return results, baseline_r2


def classify_mandatory(ablation_results: list) -> Tuple[str, Optional[float]]:
    """Classify variable as mandatory/non-causal from ablation results."""
    causal_entries = [r for r in ablation_results if r['verdict'] == 'CAUSAL']
    if not causal_entries:
        return 'NON_CAUSAL', None
    breaking_point = min(r['k_frac'] for r in causal_entries)
    if breaking_point <= 0.10:
        return 'MANDATORY_CONCENTRATED', breaking_point
    elif breaking_point <= 0.60:
        return 'MANDATORY_DISTRIBUTED', breaking_point
    else:
        return 'MANDATORY_REDUNDANT', breaking_point


# ---------------------------------------------------------------------------
# Per-architecture pipeline
# ---------------------------------------------------------------------------

def run_single_architecture(
    arch_name: str,
    H_trained: np.ndarray,
    H_untrained: np.ndarray,
    output_pred: np.ndarray,
    targets: Dict[str, np.ndarray],
    groups: np.ndarray,
    output_r2: float,
) -> dict:
    """Run probing + ablation + iAAFT for one architecture.

    Parameters
    ----------
    arch_name : str
        Architecture identifier (e.g., 'MLP_100ms', 'PySR', 'LSTM').
    H_trained : ndarray, (n_samples, n_hidden)
    H_untrained : ndarray, (n_samples, n_hidden)
    output_pred : ndarray, (n_samples,) or (n_samples, n_out)
    targets : dict mapping var_name -> ndarray (n_samples,)
    groups : ndarray, (n_samples,)
    output_r2 : float
        Output prediction R2 (for equivalent accuracy check).

    Returns
    -------
    arch_result : dict with per-variable probing, ablation, iAAFT results.
    """
    logger.info("=== Architecture: %s (output R2=%.3f) ===", arch_name,
                output_r2)

    n_samples = H_trained.shape[0]
    var_results = {}

    for var_name, target_y in targets.items():
        n = min(n_samples, len(target_y))
        if n < 20:
            logger.warning("  %s: too few samples (%d), skipping", var_name, n)
            continue

        H_tr = H_trained[:n]
        H_un = H_untrained[:n]
        y = target_y[:n]
        g = groups[:n]
        out_p = output_pred[:n] if output_pred.shape[0] >= n else output_pred

        # 1. Probing: Ridge delta-R2 with GroupKFold
        probe_result = ridge_delta_r2_groupkfold(H_tr, H_un, y, g)
        delta_r2 = probe_result['delta_R2']

        # 2. iAAFT null
        iaaft_r2, iaaft_null, iaaft_p = iaaft_null_distribution(
            H_tr, y, g, n_surrogates=min(IAAFT_N_SURROGATES, 50),
        )

        # 3. Ablation (only if delta_R2 > threshold)
        ablation_result = None
        classification = 'ZOMBIE'
        breaking_point = None

        if delta_r2 > DELTA_THRESHOLD and iaaft_p < 0.05:
            abl_results, abl_baseline = resample_ablation_generic(
                H_tr, y, out_p, g,
            )
            classification, breaking_point = classify_mandatory(abl_results)
            ablation_result = {
                'baseline_r2': abl_baseline,
                'steps': abl_results,
                'classification': classification,
                'breaking_point': breaking_point,
            }
        elif delta_r2 > DELTA_THRESHOLD:
            classification = 'LEARNED_NOT_TEMPORAL'
        else:
            classification = 'ZOMBIE'

        var_results[var_name] = {
            'delta_R2': float(delta_r2),
            'R2_trained': probe_result['R2_trained'],
            'R2_untrained': probe_result['R2_untrained'],
            'iaaft_p_value': float(iaaft_p),
            'iaaft_observed_r2': float(iaaft_r2),
            'classification': classification,
            'breaking_point': breaking_point,
            'ablation': ablation_result,
        }

        logger.info("  %s: dR2=%.3f  iAAFT_p=%.3f  class=%s",
                     var_name, delta_r2, iaaft_p, classification)

    return {
        'architecture': arch_name,
        'output_r2': float(output_r2),
        'n_hidden': int(H_trained.shape[1]),
        'variables': var_results,
    }


# ---------------------------------------------------------------------------
# MLP training + extraction helper
# ---------------------------------------------------------------------------

def _train_mlp_architecture(
    X_sequence: np.ndarray,
    Y_sequence: np.ndarray,
    window_ms: int,
    dt_ms: float = 0.5,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train MLP with specified window and extract hidden states.

    Returns (H_trained, H_untrained, output_pred, test_r2).
    """
    import torch
    from descartes.council_controls.priority1_architecture.mlp_timelag import (
        MLPTimeLag, create_windowed_dataset, train_mlp_timelag,
    )

    window_bins = max(1, int(window_ms / dt_ms))
    n_input = X_sequence.shape[1]
    n_output = Y_sequence.shape[1] if Y_sequence.ndim > 1 else 1
    if Y_sequence.ndim == 1:
        Y_sequence = Y_sequence.reshape(-1, 1)

    # Create windowed dataset
    X_w, Y_a = create_windowed_dataset(X_sequence, Y_sequence, window_bins)
    n_samples = X_w.shape[0]

    # Train/test split
    split = int(0.8 * n_samples)
    train_mask = np.zeros(n_samples, dtype=bool)
    train_mask[:split] = True
    test_mask = ~train_mask

    # Trained model
    model_trained = MLPTimeLag(n_input, n_output, window_bins)
    model_trained, cc_trained, _ = train_mlp_timelag(
        model_trained, X_w, Y_a, train_mask, test_mask, device=device,
    )

    # Extract trained hidden states
    model_trained.train(False)
    H_trained = model_trained.extract_hidden_states(X_sequence)

    # Untrained model (random init)
    model_untrained = MLPTimeLag(n_input, n_output, window_bins)
    model_untrained.train(False)
    H_untrained = model_untrained.extract_hidden_states(X_sequence)

    # Predictions on full sequence for ablation
    X_w_tensor = torch.tensor(X_w, dtype=torch.float32)
    with torch.no_grad():
        output_pred = model_trained(X_w_tensor.to(device)).cpu().numpy()

    # Test R2
    with torch.no_grad():
        X_test = torch.tensor(X_w[test_mask], dtype=torch.float32).to(device)
        pred_test = model_trained(X_test).cpu().numpy().ravel()
    true_test = Y_a[test_mask].ravel()
    ss_res = np.sum((pred_test - true_test) ** 2)
    ss_tot = np.sum((true_test - true_test.mean()) ** 2)
    test_r2 = 1 - ss_res / max(ss_tot, 1e-10)

    # Align sizes
    n_aligned = min(H_trained.shape[0], H_untrained.shape[0], n_samples)
    return (H_trained[:n_aligned], H_untrained[:n_aligned],
            output_pred[:n_aligned], float(test_r2))


def _train_pysr_architecture(
    X_sequence: np.ndarray,
    Y_sequence: np.ndarray,
    window_bins: int = 100,
    lag_step: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train PySR/gplearn surrogate and extract hidden states.

    Returns (H_trained, H_untrained, output_pred, test_r2).
    """
    from descartes.council_controls.priority1_architecture.symbolic_regression import (
        fit_and_extract,
    )

    result = fit_and_extract(
        X_sequence, Y_sequence, window_bins=window_bins, lag_step=lag_step,
    )

    H_trained = result['hidden_trained']
    H_untrained = result['hidden_untrained']
    output_pred = result['predictions']
    test_r2 = result['test_r2']

    # Align H to output_pred length
    n = min(H_trained.shape[0], H_untrained.shape[0])
    return H_trained[:n], H_untrained[:n], output_pred, test_r2


# ---------------------------------------------------------------------------
# Comparison table and decision logic
# ---------------------------------------------------------------------------

def build_comparison_table(arch_results: List[dict]) -> dict:
    """Build Variable x Architecture comparison table.

    Parameters
    ----------
    arch_results : list of dicts from run_single_architecture()

    Returns
    -------
    table : dict with keys:
        'variables': dict mapping var_name -> dict mapping arch -> classification
        'architecture_invariant': list of var names mandatory in >= 2 non-LSTM
        'lstm_specific': list of var names mandatory ONLY in LSTM
        'decision': 'PASS' or 'FAIL'
        'reasoning': str
    """
    # Collect all variables across architectures
    all_vars = set()
    for ar in arch_results:
        all_vars.update(ar['variables'].keys())

    # Build table: var_name -> arch_name -> classification
    table = {}
    for var in sorted(all_vars):
        table[var] = {}
        for ar in arch_results:
            arch = ar['architecture']
            vr = ar['variables'].get(var, {})
            table[var][arch] = vr.get('classification', 'NOT_TESTED')

    # Decision logic
    non_lstm_archs = [a for a in ARCHITECTURE_NAMES if a != 'LSTM']

    architecture_invariant = []
    lstm_specific = []

    for var in sorted(all_vars):
        is_mandatory_lstm = _is_mandatory(table[var].get('LSTM', 'NOT_TESTED'))

        n_mandatory_non_lstm = sum(
            1 for arch in non_lstm_archs
            if _is_mandatory(table[var].get(arch, 'NOT_TESTED'))
        )

        if n_mandatory_non_lstm >= 2:
            architecture_invariant.append(var)
        elif is_mandatory_lstm and n_mandatory_non_lstm == 0:
            lstm_specific.append(var)

    # Collect all mandatory in LSTM
    mandatory_in_lstm = [
        var for var in all_vars
        if _is_mandatory(table[var].get('LSTM', 'NOT_TESTED'))
    ]

    # Decision
    if len(architecture_invariant) > 0:
        decision = 'PASS'
        reasoning = (
            f"{len(architecture_invariant)} variable(s) are architecture-"
            f"invariant (mandatory in >= 2 non-LSTM architectures): "
            f"{architecture_invariant}. This supports genuine biophysical "
            f"encoding rather than LSTM-specific artifacts."
        )
    elif len(mandatory_in_lstm) > 0 and all(
        v in lstm_specific for v in mandatory_in_lstm
    ):
        decision = 'FAIL'
        reasoning = (
            f"All {len(mandatory_in_lstm)} mandatory variable(s) are "
            f"LSTM-specific (not found in any non-LSTM architecture): "
            f"{lstm_specific}. Cannot rule out LSTM architectural artifact."
        )
    else:
        decision = 'INCONCLUSIVE'
        reasoning = (
            f"Mixed results: {len(architecture_invariant)} architecture-"
            f"invariant, {len(lstm_specific)} LSTM-specific. "
            f"Insufficient evidence for definitive PASS or FAIL."
        )

    return {
        'variables': table,
        'architecture_invariant': architecture_invariant,
        'lstm_specific': lstm_specific,
        'decision': decision,
        'reasoning': reasoning,
    }


def _is_mandatory(classification: str) -> bool:
    """Check if a classification indicates mandatory status."""
    return classification in (
        'MANDATORY_CONCENTRATED',
        'MANDATORY_DISTRIBUTED',
        'MANDATORY_REDUNDANT',
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_p1_control(
    circuit_data: Dict[str, dict],
    targets: Dict[str, np.ndarray],
    results_dir: str = 'results/council_controls',
    dt_ms: float = 0.5,
    device: str = 'cpu',
    include_lstm: bool = True,
    lstm_hidden_trained: Optional[np.ndarray] = None,
    lstm_hidden_untrained: Optional[np.ndarray] = None,
    lstm_output_pred: Optional[np.ndarray] = None,
    lstm_output_r2: float = 0.0,
) -> dict:
    """Run full P1 architecture control.

    Parameters
    ----------
    circuit_data : dict mapping circuit_name -> {
        'X': ndarray (T, n_input),  -- input sequences
        'Y': ndarray (T, n_output), -- output sequences
    }
    targets : dict mapping var_name -> ndarray (n_samples,)
        Biophysical target variables for probing.
    results_dir : str
    dt_ms : float
    device : str
    include_lstm : bool
        If True and LSTM hidden states are provided, include LSTM in comparison.
    lstm_hidden_trained, lstm_hidden_untrained, lstm_output_pred : ndarray
        Pre-computed LSTM hidden states (optional).
    lstm_output_r2 : float

    Returns
    -------
    control_result : dict with full comparison table and decision.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_arch_results = []

    for circuit_name, cdata in circuit_data.items():
        logger.info("=" * 60)
        logger.info("Circuit: %s", circuit_name)
        logger.info("=" * 60)

        X_seq = cdata['X']
        Y_seq = cdata['Y']
        T = X_seq.shape[0]

        # Create group labels (trial groups for GroupKFold)
        # If no trial structure, use time-based blocks
        trial_size = cdata.get('trial_size', 2000)
        n_trials = max(1, T // trial_size)
        groups = np.repeat(np.arange(n_trials), trial_size)[:T]

        # --- MLP architectures ---
        for window_ms in MLP_WINDOW_SIZES_MS:
            arch_name = f"MLP_{window_ms}ms"
            logger.info("Training %s...", arch_name)
            try:
                H_tr, H_un, out_pred, test_r2 = _train_mlp_architecture(
                    X_seq, Y_seq, window_ms, dt_ms, device,
                )
                # Align groups
                n = min(H_tr.shape[0], len(groups))
                g = groups[:n]

                # Align targets
                aligned_targets = {}
                for vn, vy in targets.items():
                    if len(vy) >= n:
                        aligned_targets[vn] = vy[:n]

                result = run_single_architecture(
                    arch_name, H_tr[:n], H_un[:n], out_pred[:n],
                    aligned_targets, g, test_r2,
                )
                result['circuit'] = circuit_name
                all_arch_results.append(result)

            except Exception as e:
                logger.error("  %s failed: %s", arch_name, e)
                continue

        # --- PySR architecture ---
        logger.info("Training PySR...")
        try:
            H_tr, H_un, out_pred, test_r2 = _train_pysr_architecture(
                X_seq, Y_seq,
            )
            n = min(H_tr.shape[0], len(groups))
            g = groups[:n]

            aligned_targets = {}
            for vn, vy in targets.items():
                if len(vy) >= n:
                    aligned_targets[vn] = vy[:n]

            result = run_single_architecture(
                'PySR', H_tr[:n], H_un[:n], out_pred[:min(out_pred.shape[0], n)],
                aligned_targets, g, test_r2,
            )
            result['circuit'] = circuit_name
            all_arch_results.append(result)

        except Exception as e:
            logger.error("  PySR failed: %s", e)

        # --- LSTM (if provided) ---
        if include_lstm and lstm_hidden_trained is not None:
            logger.info("Including LSTM baseline...")
            n = min(lstm_hidden_trained.shape[0], len(groups))
            g = groups[:n]

            aligned_targets = {}
            for vn, vy in targets.items():
                if len(vy) >= n:
                    aligned_targets[vn] = vy[:n]

            result = run_single_architecture(
                'LSTM', lstm_hidden_trained[:n], lstm_hidden_untrained[:n],
                lstm_output_pred[:n], aligned_targets, g, lstm_output_r2,
            )
            result['circuit'] = circuit_name
            all_arch_results.append(result)

    # Build comparison table
    comparison = build_comparison_table(all_arch_results)

    # Assemble full output
    output = {
        'control': 'P1_ARCHITECTURE',
        'n_architectures': len(all_arch_results),
        'architectures_tested': [ar['architecture'] for ar in all_arch_results],
        'comparison_table': comparison['variables'],
        'architecture_invariant': comparison['architecture_invariant'],
        'lstm_specific': comparison['lstm_specific'],
        'decision': comparison['decision'],
        'reasoning': comparison['reasoning'],
        'per_architecture': all_arch_results,
    }

    # Save
    save_path = results_dir / 'p1_architecture_control.json'

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)

    logger.info("=" * 60)
    logger.info("P1 ARCHITECTURE CONTROL: %s", output['decision'])
    logger.info("  Architecture-invariant: %s",
                output['architecture_invariant'])
    logger.info("  LSTM-specific: %s", output['lstm_specific'])
    logger.info("  Reasoning: %s", output['reasoning'])
    logger.info("Saved to %s", save_path)
    logger.info("=" * 60)

    return output
