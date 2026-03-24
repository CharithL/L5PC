"""
arbitrary_targets.py

Priority 2: Arbitrary Probe Target Generator

Generates three types of arbitrary (non-biophysical) probe targets to test
specificity of probing results. If a model's hidden states decode arbitrary
targets as well as biophysical ones, the probing is not specific.

Three target types:
  1. Random linear projections of hidden states
     - Random matrix W (n_hidden, k) projects hidden states to k targets
     - These are trivially decodable from hidden states (by construction)
     - If biophysical probes do no better than these, probing is not specific

  2. PCA components of the input signal
     - Top PCA components of the raw input (not hidden states)
     - These reflect input structure, not learned representations
     - High probing R2 for these would indicate input-passthrough

  3. Unrelated-domain variables
     - Synthetic time series from unrelated domains (Lorenz, financial, etc.)
     - These have realistic temporal structure but zero causal relationship
     - Any probing success is purely spurious

No external dependencies beyond numpy, scipy, sklearn.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type 1: Random linear projections of hidden states
# ---------------------------------------------------------------------------

def generate_random_projections(
    H: np.ndarray,
    n_targets: int = 10,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate random linear projection targets from hidden states.

    These targets are trivially decodable from the hidden states by
    construction (they are linear functions of H). Any probe should
    decode these with high R2. The question is whether BIOPHYSICAL
    targets achieve DIFFERENT (higher or lower) R2 than these controls.

    Parameters
    ----------
    H : ndarray, (n_samples, n_hidden)
        Hidden state matrix.
    n_targets : int
        Number of random projection targets to generate.
    random_state : int

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    n_hidden = H.shape[1]

    targets = {}
    for i in range(n_targets):
        # Random projection vector (unit normalized)
        w = rng.randn(n_hidden)
        w = w / (np.linalg.norm(w) + 1e-10)

        # Project hidden states
        projection = H @ w

        # Add small noise to prevent perfect R2
        noise = rng.randn(len(projection)) * 0.01 * np.std(projection)
        targets[f'random_proj_{i}'] = projection + noise

    return targets


# ---------------------------------------------------------------------------
# Type 2: PCA components of the input signal
# ---------------------------------------------------------------------------

def generate_pca_input_targets(
    X_input: np.ndarray,
    n_components: int = 5,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate PCA component targets from the raw input signal.

    These targets capture the dominant modes of input variation.
    If hidden states decode these well, it indicates the model is
    primarily passing through input structure rather than computing
    novel representations.

    Parameters
    ----------
    X_input : ndarray, (n_samples, n_input)
        Raw input features (NOT hidden states).
    n_components : int
        Number of PCA components to extract.
    random_state : int

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_samples,)
    """
    _eps = 1e-8
    X_scaled = (X_input - X_input.mean(axis=0)) / (X_input.std(axis=0) + _eps)

    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    components = pca.fit_transform(X_scaled)

    targets = {}
    for i in range(n_comp):
        targets[f'pca_input_{i}'] = components[:, i]
        logger.debug("  PCA component %d: explained_var=%.3f",
                     i, pca.explained_variance_ratio_[i])

    return targets


# ---------------------------------------------------------------------------
# Type 3: Unrelated-domain variables
# ---------------------------------------------------------------------------

def _lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    """Lorenz attractor ODEs."""
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def generate_lorenz_targets(
    n_samples: int,
    dt: float = 0.01,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate Lorenz attractor time series as unrelated-domain targets.

    The Lorenz system has realistic temporal structure (deterministic chaos)
    but zero causal relationship with neural data.

    Parameters
    ----------
    n_samples : int
    dt : float
    random_state : int

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    initial = rng.randn(3) * 5 + np.array([1.0, 1.0, 1.0])

    t_span = (0, n_samples * dt)
    t_points = np.linspace(0, n_samples * dt, n_samples)

    sol = solve_ivp(
        _lorenz_system, t_span, initial, t_eval=t_points,
        method='RK45', max_step=dt,
    )

    if sol.success and sol.y.shape[1] >= n_samples:
        targets = {
            'lorenz_x': sol.y[0, :n_samples],
            'lorenz_y': sol.y[1, :n_samples],
            'lorenz_z': sol.y[2, :n_samples],
        }
    else:
        # Fallback: simple chaotic map
        logger.warning("Lorenz integration failed, using logistic map fallback")
        targets = _logistic_map_targets(n_samples, rng)

    return targets


def _logistic_map_targets(n_samples, rng):
    """Logistic map as a fallback chaotic system."""
    r_values = [3.7, 3.8, 3.9]
    targets = {}
    for i, r in enumerate(r_values):
        x = rng.uniform(0.1, 0.9)
        series = np.zeros(n_samples)
        for t in range(n_samples):
            series[t] = x
            x = r * x * (1 - x)
        targets[f'logistic_r{r}'] = series
    return targets


def generate_random_walk_targets(
    n_samples: int,
    n_targets: int = 3,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate correlated random walk targets (financial-like).

    Parameters
    ----------
    n_samples : int
    n_targets : int
    random_state : int

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    targets = {}

    for i in range(n_targets):
        # Geometric Brownian motion-like
        mu = rng.uniform(-0.01, 0.01)
        sigma = rng.uniform(0.01, 0.1)
        increments = rng.normal(mu, sigma, n_samples)
        series = np.cumsum(increments)
        targets[f'random_walk_{i}'] = series

    return targets


def generate_sinusoidal_targets(
    n_samples: int,
    dt: float = 0.5,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate sinusoidal mixtures as unrelated targets.

    These have clear temporal structure but no neural relevance.
    """
    rng = np.random.RandomState(random_state)
    t = np.arange(n_samples) * dt / 1000.0  # Convert to seconds

    targets = {}
    for i in range(3):
        freq = rng.uniform(1, 50)  # Hz
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        signal = amp * np.sin(2 * np.pi * freq * t + phase)

        # Add harmonics
        for h in range(2, 4):
            harm_amp = rng.uniform(0.1, 0.5) * amp
            signal += harm_amp * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2 * np.pi))

        targets[f'sinusoidal_mix_{i}'] = signal

    return targets


# ---------------------------------------------------------------------------
# Unified arbitrary target generator
# ---------------------------------------------------------------------------

def generate_all_arbitrary_targets(
    H: np.ndarray,
    X_input: np.ndarray,
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate all three types of arbitrary probe targets.

    Parameters
    ----------
    H : ndarray, (n_samples, n_hidden)
        Hidden state matrix (for random projections).
    X_input : ndarray, (n_samples, n_input)
        Raw input signal (for PCA components).
    n_samples : int, optional
        Override sample count.
    random_state : int

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_samples,)
        All arbitrary targets, with type prefix in name:
        'random_proj_*', 'pca_input_*', 'lorenz_*', 'random_walk_*',
        'sinusoidal_mix_*', 'logistic_*'
    """
    if n_samples is None:
        n_samples = H.shape[0]

    all_targets = {}

    # Type 1: Random projections
    rp_targets = generate_random_projections(
        H[:n_samples], n_targets=10, random_state=random_state,
    )
    all_targets.update(rp_targets)

    # Type 2: PCA input components
    pca_targets = generate_pca_input_targets(
        X_input[:n_samples], n_components=5, random_state=random_state,
    )
    all_targets.update(pca_targets)

    # Type 3: Unrelated-domain variables
    lorenz_targets = generate_lorenz_targets(
        n_samples, random_state=random_state,
    )
    all_targets.update(lorenz_targets)

    rw_targets = generate_random_walk_targets(
        n_samples, n_targets=3, random_state=random_state,
    )
    all_targets.update(rw_targets)

    sin_targets = generate_sinusoidal_targets(
        n_samples, random_state=random_state,
    )
    all_targets.update(sin_targets)

    logger.info("Generated %d arbitrary targets: %d random_proj, %d pca, "
                "%d lorenz, %d random_walk, %d sinusoidal",
                len(all_targets), len(rp_targets), len(pca_targets),
                len(lorenz_targets), len(rw_targets), len(sin_targets))

    return all_targets


def get_target_type(target_name: str) -> str:
    """Classify an arbitrary target by its type from the name prefix."""
    if target_name.startswith('random_proj'):
        return 'random_projection'
    elif target_name.startswith('pca_input'):
        return 'pca_input'
    elif target_name.startswith('lorenz') or target_name.startswith('logistic'):
        return 'unrelated_chaotic'
    elif target_name.startswith('random_walk'):
        return 'unrelated_random_walk'
    elif target_name.startswith('sinusoidal'):
        return 'unrelated_sinusoidal'
    else:
        return 'unknown'
