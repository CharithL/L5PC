#!/usr/bin/env python3
"""
End-to-end test of P1 Architecture Control pipeline.
Uses synthetic data to verify the full pipeline runs without errors.
"""

import sys
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('test_p1_e2e')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_circuit(n_trials=20, trial_len=200, n_in=10, n_out=3, seed=42):
    """Generate synthetic circuit data with known structure."""
    rng = np.random.default_rng(seed)

    # Input: random with some oscillatory structure
    t = np.arange(trial_len * n_trials) * 0.01  # 10ms bins
    X = rng.normal(0, 1, (len(t), n_in)).astype(np.float32)
    # Add theta oscillation to first 3 neurons
    for i in range(3):
        X[:, i] += 0.5 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2*np.pi))

    # Output: nonlinear function of input (so there's something to learn)
    Y = np.tanh(X[:, :n_out] @ rng.normal(0, 0.5, (n_out, n_out))).astype(np.float32)
    Y += rng.normal(0, 0.1, Y.shape).astype(np.float32)

    # Bio targets
    pop_rate = np.mean(X, axis=1)
    targets = {
        'firing_rate_input': pop_rate,
        'theta_power': np.abs(np.sin(2 * np.pi * 6 * t)) ** 2,
        'gamma_power': np.abs(np.sin(2 * np.pi * 40 * t)) ** 2,
        'theta_gamma_pac': np.abs(np.sin(2 * np.pi * 6 * t)) * np.abs(np.sin(2 * np.pi * 40 * t)),
        'trial_variance': np.var(X, axis=1),
    }

    circuit_data = {
        'test_circuit': {
            'X': X, 'Y': Y, 'trial_size': trial_len,
        }
    }

    return circuit_data, targets


def test_epsilon_safe_scaling():
    """Test that epsilon-safe scaling handles zero-variance features."""
    log.info("--- Test 1: Epsilon-safe scaling ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        _safe_scale_fit, _safe_scale_transform)

    # Normal data
    X = np.random.randn(100, 10)
    mean, std = _safe_scale_fit(X)
    X_scaled = _safe_scale_transform(X, mean, std)
    assert not np.any(np.isnan(X_scaled)), "NaN in normal scaling"
    log.info("  Normal data: OK")

    # Data with zero-variance columns (the bug case)
    X_degenerate = np.random.randn(100, 10)
    X_degenerate[:, 3] = 5.0  # constant column
    X_degenerate[:, 7] = 0.0  # zero column
    mean, std = _safe_scale_fit(X_degenerate)
    X_scaled = _safe_scale_transform(X_degenerate, mean, std)
    assert not np.any(np.isnan(X_scaled)), "NaN in degenerate scaling!"
    assert not np.any(np.isinf(X_scaled)), "Inf in degenerate scaling!"
    log.info("  Zero-variance columns: OK (no NaN/Inf)")

    # All-zero data
    X_zero = np.zeros((100, 10))
    mean, std = _safe_scale_fit(X_zero)
    X_scaled = _safe_scale_transform(X_zero, mean, std)
    assert not np.any(np.isnan(X_scaled)), "NaN in all-zero scaling!"
    log.info("  All-zero data: OK")

    log.info("  PASSED")


def test_ridge_delta_r2():
    """Test Ridge delta-R2 probing with GroupKFold."""
    log.info("--- Test 2: Ridge delta-R2 with GroupKFold ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        ridge_delta_r2_groupkfold)

    rng = np.random.default_rng(42)
    n_samples = 2000
    n_hidden = 32
    n_trials = 20
    trial_len = n_samples // n_trials

    # Hidden states: trained has signal, untrained is random
    target = np.sin(np.linspace(0, 10, n_samples))
    H_trained = rng.normal(0, 1, (n_samples, n_hidden))
    # Inject signal into first 5 dims
    for i in range(5):
        H_trained[:, i] += target * (0.5 + 0.1 * i)

    H_untrained = rng.normal(0, 1, (n_samples, n_hidden))
    # Add zero-variance columns to untrained (simulating dead neurons)
    H_untrained[:, 10] = 0.0
    H_untrained[:, 15] = 3.14
    H_untrained[:, 20] = 0.0

    groups = np.repeat(np.arange(n_trials), trial_len)

    result = ridge_delta_r2_groupkfold(H_trained, H_untrained, target, groups)

    log.info("  R2_trained: %.4f", result['R2_trained'])
    log.info("  R2_untrained: %.4f", result['R2_untrained'])
    log.info("  delta_R2: %.4f", result['delta_R2'])

    assert not np.isnan(result['delta_R2']), "delta_R2 is NaN!"
    assert result['delta_R2'] > 0, "delta_R2 should be positive (trained has signal)"
    assert result['R2_trained'] > result['R2_untrained'], "Trained should beat untrained"

    log.info("  PASSED (dR2=%.4f > 0)", result['delta_R2'])


def test_iaaft():
    """Test iAAFT null distribution."""
    log.info("--- Test 3: iAAFT null distribution ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        iaaft_null_distribution)

    rng = np.random.default_rng(42)
    n_samples = 1000
    n_hidden = 16
    n_trials = 10
    trial_len = n_samples // n_trials

    target = np.sin(np.linspace(0, 10, n_samples))
    H = rng.normal(0, 1, (n_samples, n_hidden))
    for i in range(3):
        H[:, i] += target * 0.5

    # Add dead neurons
    H[:, 8] = 0.0
    H[:, 12] = 1.0

    groups = np.repeat(np.arange(n_trials), trial_len)

    # Use fewer surrogates for speed
    real_r2, surr_r2s, p_value = iaaft_null_distribution(
        H, target, groups, n_surrogates=10)

    log.info("  real_R2: %.4f", real_r2)
    log.info("  surr_mean: %.4f", np.mean(surr_r2s))
    log.info("  p_value: %.4f", p_value)

    assert not np.isnan(real_r2), "real_R2 is NaN!"
    assert not any(np.isnan(surr_r2s)), "surrogate R2s contain NaN!"

    log.info("  PASSED")


def test_resample_ablation():
    """Test resample ablation."""
    log.info("--- Test 4: Resample ablation ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        resample_ablation_generic)

    rng = np.random.default_rng(42)
    n_samples = 1000
    n_hidden = 16

    H = rng.normal(0, 1, (n_samples, n_hidden))
    target = H[:, 0] * 2 + H[:, 1] * 1.5 + rng.normal(0, 0.1, n_samples)
    output = H @ rng.normal(0, 0.3, (n_hidden, 3))

    # Add dead neurons
    H[:, 8] = 0.0

    groups = np.repeat(np.arange(10), 100)
    results_list, baseline_r2 = resample_ablation_generic(H, target, output, groups)

    log.info("  baseline_R2: %.4f", baseline_r2)
    log.info("  n_k_tested: %d", len(results_list))
    for k_info in results_list:
        log.info("    k_frac=%.2f: z=%.2f verdict=%s",
                 k_info['k_frac'], k_info['z_score'], k_info['verdict'])

    assert not np.isnan(baseline_r2), "baseline_R2 is NaN!"

    log.info("  PASSED")


def test_full_single_architecture():
    """Test run_single_architecture end-to-end."""
    log.info("--- Test 5: Full single architecture pipeline ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        run_single_architecture)

    rng = np.random.default_rng(42)
    n_samples = 2000
    n_hidden = 32
    n_trials = 20
    trial_len = n_samples // n_trials

    target = np.sin(np.linspace(0, 10, n_samples))
    H_trained = rng.normal(0, 1, (n_samples, n_hidden))
    for i in range(5):
        H_trained[:, i] += target * 0.5

    H_untrained = rng.normal(0, 1, (n_samples, n_hidden))
    # Dead neurons in untrained
    H_untrained[:, 10:15] = 0.0

    groups = np.repeat(np.arange(n_trials), trial_len)
    output_pred = H_trained @ rng.normal(0, 0.1, (n_hidden, 3))

    targets = {
        'firing_rate': target,
        'theta_power': np.abs(np.sin(np.linspace(0, 20, n_samples))),
    }

    result = run_single_architecture(
        'test_MLP', H_trained, H_untrained, output_pred,
        targets, groups, output_r2=0.5)

    log.info("  Architecture: %s", result['architecture'])
    log.info("  Variables probed: %d", len(result['variables']))
    for vname, vresult in result['variables'].items():
        log.info("    %s: dR2=%.4f  class=%s",
                 vname, vresult['delta_R2'], vresult['classification'])
        assert not np.isnan(vresult['delta_R2']), f"NaN in {vname} delta_R2!"

    log.info("  PASSED")


def test_full_p1_pipeline():
    """Test the complete run_p1_control pipeline with synthetic data."""
    log.info("--- Test 6: Full P1 pipeline (synthetic data) ---")
    from descartes.council_controls.priority1_architecture.run_p1_control import (
        run_p1_control)

    circuit_data, targets = generate_synthetic_circuit(
        n_trials=15, trial_len=150, n_in=8, n_out=3)

    # Generate fake LSTM hidden states
    rng = np.random.default_rng(42)
    T = circuit_data['test_circuit']['X'].shape[0]
    lstm_h_tr = rng.normal(0, 1, (T, 32)).astype(np.float64)
    # Inject some signal
    for i, (vname, vtarget) in enumerate(targets.items()):
        if i < 5:
            lstm_h_tr[:, i] += vtarget[:T] * 0.3

    lstm_h_un = rng.normal(0, 1, (T, 32)).astype(np.float64)
    lstm_h_un[:, 10:15] = 0.0  # dead neurons
    lstm_pred = rng.normal(0, 0.1, (T, 3))

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_p1_control(
            circuit_data=circuit_data,
            targets=targets,
            results_dir=tmpdir,
            dt_ms=10.0,
            device='cpu',
            include_lstm=True,
            lstm_hidden_trained=lstm_h_tr,
            lstm_hidden_untrained=lstm_h_un,
            lstm_output_pred=lstm_pred,
            lstm_output_r2=0.5,
        )

    log.info("  Decision: %s", result['decision'])
    log.info("  Architectures tested: %d", result['n_architectures'])
    log.info("  Architecture-invariant: %s", result['architecture_invariant'])
    log.info("  LSTM-specific: %s", result['lstm_specific'])
    log.info("  Reasoning: %s", result['reasoning'])

    assert result['decision'] in ('PASS', 'FAIL', 'INCONCLUSIVE'), \
        f"Unexpected decision: {result['decision']}"
    assert result['n_architectures'] > 0, "No architectures tested!"

    log.info("  PASSED")


def main():
    log.info("=" * 60)
    log.info("P1 ARCHITECTURE CONTROL -- END-TO-END TEST")
    log.info("=" * 60)

    tests = [
        ("Epsilon-safe scaling", test_epsilon_safe_scaling),
        ("Ridge delta-R2", test_ridge_delta_r2),
        ("iAAFT null", test_iaaft),
        ("Resample ablation", test_resample_ablation),
        ("Single architecture", test_full_single_architecture),
        ("Full P1 pipeline", test_full_p1_pipeline),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            log.error("  FAILED: %s -- %s", name, e)
            import traceback
            traceback.print_exc()

    log.info("\n" + "=" * 60)
    log.info("RESULTS: %d/%d passed, %d failed", passed, len(tests), failed)
    if errors:
        for name, err in errors:
            log.error("  FAIL: %s -- %s", name, err)
    else:
        log.info("ALL TESTS PASSED")
    log.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
