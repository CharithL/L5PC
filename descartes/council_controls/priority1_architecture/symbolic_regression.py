"""
symbolic_regression.py

Phase 1B: Symbolic Regression Surrogate (PySR / gplearn fallback)

NON-DIFFERENTIABLE, NON-CONTINUOUS control architecture for DESCARTES.

Key properties distinguishing this from LSTM:
- NO gradient-based learning (genetic programming / symbolic search)
- NO hidden state memory (each prediction is a pure function of inputs)
- NO oscillatory dynamics possible (closed-form algebraic expression)
- Expression complexity is bounded and human-readable

Protocol per output neuron:
  1. Construct time-lagged feature matrix from input spike trains
  2. Run PySR symbolic regression (or gplearn fallback)
  3. Select Pareto-optimal expression (accuracy vs complexity)
  4. "Hidden state" analog = sub-expression values at each timestep
  5. Untrained baseline = sub-expression values from a random expression

No external dependencies beyond numpy, scipy, sklearn.
PySR and gplearn are optional -- gplearn is used as fallback, and if
neither is available, a minimal GP implementation is provided.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-lagged feature construction
# ---------------------------------------------------------------------------

def build_timelag_features(X_sequence: np.ndarray, window_bins: int,
                           lag_step: int = 1) -> np.ndarray:
    """Construct time-lagged feature matrix for symbolic regression.

    Parameters
    ----------
    X_sequence : ndarray, shape (T, n_input)
        Full input time series.
    window_bins : int
        Number of past timesteps to include.
    lag_step : int
        Stride between lags (1 = every bin, 2 = every other, etc.).

    Returns
    -------
    X_lagged : ndarray, shape (T - window_bins + 1, n_input * n_lags)
        Flattened time-lagged features at each valid timestep.
    """
    T, n_input = X_sequence.shape
    lag_indices = list(range(0, window_bins, lag_step))
    n_lags = len(lag_indices)
    n_samples = T - window_bins + 1

    if n_samples < 1:
        return np.zeros((0, n_input * n_lags))

    X_lagged = np.zeros((n_samples, n_input * n_lags), dtype=np.float32)
    for t in range(n_samples):
        for j, lag in enumerate(lag_indices):
            idx = t + window_bins - 1 - lag
            X_lagged[t, j * n_input:(j + 1) * n_input] = X_sequence[idx]
    return X_lagged


# ---------------------------------------------------------------------------
# Expression tree representation (minimal GP for fallback)
# ---------------------------------------------------------------------------

_UNARY_OPS = {
    'neg': lambda x: -x,
    'abs': lambda x: np.abs(x),
    'square': lambda x: x ** 2,
    'sqrt_abs': lambda x: np.sqrt(np.abs(x) + 1e-10),
    'log_abs': lambda x: np.log(np.abs(x) + 1e-10),
    'sin': lambda x: np.sin(x),
    'tanh': lambda x: np.tanh(x),
}

_BINARY_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / (b + np.sign(b) * 1e-10 + 1e-10 * (b == 0)),
}


class ExprNode:
    """Node in a symbolic expression tree."""

    def __init__(self, op=None, left=None, right=None,
                 feature_idx=None, constant=None):
        self.op = op
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.constant = constant

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate expression on feature matrix X (n_samples, n_features)."""
        if self.feature_idx is not None:
            return X[:, self.feature_idx].copy()
        if self.constant is not None:
            return np.full(X.shape[0], self.constant, dtype=np.float64)
        if self.op in _UNARY_OPS:
            child_val = self.left.evaluate(X)
            return _UNARY_OPS[self.op](child_val)
        if self.op in _BINARY_OPS:
            left_val = self.left.evaluate(X)
            right_val = self.right.evaluate(X)
            return _BINARY_OPS[self.op](left_val, right_val)
        raise ValueError(f"Unknown op: {self.op}")

    def extract_subexpressions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Recursively evaluate and collect all sub-expression values.

        Returns dict mapping sub-expression string -> (n_samples,) array.
        This is the 'hidden state' analog for probing.
        """
        results = {}
        self._collect(X, results, depth=0, path="root")
        return results

    def _collect(self, X, results, depth, path):
        val = self.evaluate(X)
        # Clip to prevent inf propagation
        val = np.clip(val, -1e6, 1e6)
        results[path] = val

        if self.left is not None:
            self.left._collect(X, results, depth + 1, f"{path}_L")
        if self.right is not None:
            self.right._collect(X, results, depth + 1, f"{path}_R")

    def complexity(self) -> int:
        """Count total nodes in tree."""
        c = 1
        if self.left:
            c += self.left.complexity()
        if self.right:
            c += self.right.complexity()
        return c

    def __repr__(self):
        if self.feature_idx is not None:
            return f"x{self.feature_idx}"
        if self.constant is not None:
            return f"{self.constant:.3f}"
        if self.op in _UNARY_OPS:
            return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"


# ---------------------------------------------------------------------------
# Random expression generation (for untrained baseline)
# ---------------------------------------------------------------------------

def random_expression(n_features: int, max_depth: int = 4,
                      rng: np.random.RandomState = None) -> ExprNode:
    """Generate a random symbolic expression tree.

    Used for the untrained baseline: sub-expression values from a random
    (untrained) expression serve as the control for probing.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    return _grow_random(n_features, max_depth, 0, rng)


def _grow_random(n_features, max_depth, depth, rng):
    if depth >= max_depth or (depth > 1 and rng.random() < 0.3):
        # Terminal: feature or constant
        if rng.random() < 0.7:
            return ExprNode(feature_idx=rng.randint(0, n_features))
        else:
            return ExprNode(constant=rng.randn())

    # Non-terminal
    if rng.random() < 0.3:
        # Unary
        op = rng.choice(list(_UNARY_OPS.keys()))
        child = _grow_random(n_features, max_depth, depth + 1, rng)
        return ExprNode(op=op, left=child)
    else:
        # Binary
        op = rng.choice(list(_BINARY_OPS.keys()))
        left = _grow_random(n_features, max_depth, depth + 1, rng)
        right = _grow_random(n_features, max_depth, depth + 1, rng)
        return ExprNode(op=op, left=left, right=right)


# ---------------------------------------------------------------------------
# Minimal genetic programming (fallback when PySR/gplearn unavailable)
# ---------------------------------------------------------------------------

class MinimalGP:
    """Bare-bones genetic programming for symbolic regression.

    This is NOT meant to compete with PySR or gplearn. It exists only
    as a fallback so the control pipeline can run without optional deps.
    """

    def __init__(self, population_size: int = 200,
                 generations: int = 50,
                 max_depth: int = 5,
                 tournament_size: int = 5,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2,
                 parsimony_coefficient: float = 0.01,
                 random_state: int = 42):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.parsimony_coefficient = parsimony_coefficient
        self.rng = np.random.RandomState(random_state)
        self.best_program_ = None
        self.pareto_front_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Run GP to find best symbolic expression."""
        n_features = X.shape[1]

        # Initialize population
        population = [
            random_expression(n_features, self.max_depth, self.rng)
            for _ in range(self.population_size)
        ]

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_list = []
            for expr in population:
                mse, complexity = self._evaluate(expr, X, y)
                fitness_list.append((mse, complexity, expr))

            # Update Pareto front
            self._update_pareto(fitness_list)

            # Sort by penalized fitness
            fitness_list.sort(
                key=lambda x: x[0] + self.parsimony_coefficient * x[1]
            )

            # Elitism: keep top 10%
            elite_n = max(1, self.population_size // 10)
            new_pop = [f[2] for f in fitness_list[:elite_n]]

            # Fill rest with tournament selection + crossover/mutation
            while len(new_pop) < self.population_size:
                if self.rng.random() < self.crossover_rate:
                    p1 = self._tournament(fitness_list)
                    p2 = self._tournament(fitness_list)
                    child = self._crossover(p1, p2, n_features)
                elif self.rng.random() < self.mutation_rate:
                    parent = self._tournament(fitness_list)
                    child = self._mutate(parent, n_features)
                else:
                    child = random_expression(n_features, self.max_depth,
                                              self.rng)
                new_pop.append(child)

            population = new_pop

        # Final evaluation
        final_fitness = []
        for expr in population:
            mse, complexity = self._evaluate(expr, X, y)
            final_fitness.append((mse, complexity, expr))
        self._update_pareto(final_fitness)

        # Best = lowest penalized MSE
        final_fitness.sort(
            key=lambda x: x[0] + self.parsimony_coefficient * x[1]
        )
        self.best_program_ = final_fitness[0][2]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_program_ is None:
            raise RuntimeError("Must call fit() first")
        pred = self.best_program_.evaluate(X)
        return np.clip(np.nan_to_num(pred, nan=0.0), -1e6, 1e6)

    def _evaluate(self, expr, X, y):
        try:
            pred = expr.evaluate(X)
            pred = np.clip(np.nan_to_num(pred, nan=0.0), -1e6, 1e6)
            mse = float(np.mean((pred - y) ** 2))
        except Exception:
            mse = 1e12
        return mse, expr.complexity()

    def _tournament(self, fitness_list):
        indices = self.rng.choice(len(fitness_list), self.tournament_size,
                                  replace=False)
        best_idx = min(indices, key=lambda i: (
            fitness_list[i][0] +
            self.parsimony_coefficient * fitness_list[i][1]
        ))
        return fitness_list[best_idx][2]

    def _crossover(self, p1, p2, n_features):
        """Swap a random subtree."""
        # Simple: use p1 structure with one subtree from p2
        child = self._copy_tree(p1)
        donor = self._random_subtree(p2)
        self._replace_random_subtree(child, donor)
        return child

    def _mutate(self, parent, n_features):
        """Replace a random subtree with a new random one."""
        child = self._copy_tree(parent)
        new_sub = random_expression(n_features, max(2, self.max_depth - 2),
                                    self.rng)
        self._replace_random_subtree(child, new_sub)
        return child

    def _copy_tree(self, node):
        if node is None:
            return None
        new_node = ExprNode(
            op=node.op,
            feature_idx=node.feature_idx,
            constant=node.constant,
        )
        new_node.left = self._copy_tree(node.left)
        new_node.right = self._copy_tree(node.right)
        return new_node

    def _random_subtree(self, node):
        """Pick a random node from the tree."""
        nodes = self._collect_nodes(node)
        return self._copy_tree(nodes[self.rng.randint(len(nodes))])

    def _collect_nodes(self, node):
        result = [node]
        if node.left:
            result.extend(self._collect_nodes(node.left))
        if node.right:
            result.extend(self._collect_nodes(node.right))
        return result

    def _replace_random_subtree(self, root, replacement):
        """Replace a random leaf or subtree in root."""
        nodes = self._collect_nodes(root)
        if len(nodes) <= 1:
            root.op = replacement.op
            root.left = replacement.left
            root.right = replacement.right
            root.feature_idx = replacement.feature_idx
            root.constant = replacement.constant
            return
        # Pick a non-root node's parent
        target = nodes[self.rng.randint(1, len(nodes))]
        target.op = replacement.op
        target.left = replacement.left
        target.right = replacement.right
        target.feature_idx = replacement.feature_idx
        target.constant = replacement.constant

    def _update_pareto(self, fitness_list):
        """Maintain Pareto front of (MSE, complexity) trade-offs."""
        candidates = [(mse, c, expr) for mse, c, expr in fitness_list
                       if mse < 1e10]
        if not candidates:
            return

        # Sort by complexity, filter dominated
        candidates.sort(key=lambda x: x[1])
        front = []
        best_mse = float('inf')
        for mse, c, expr in candidates:
            if mse < best_mse:
                front.append((mse, c, expr))
                best_mse = mse

        self.pareto_front_ = front


# ---------------------------------------------------------------------------
# Unified symbolic regression interface
# ---------------------------------------------------------------------------

def fit_symbolic_surrogate(
    X_lagged: np.ndarray,
    y: np.ndarray,
    output_idx: int = 0,
    pysr_kwargs: Optional[dict] = None,
    random_state: int = 42,
) -> Tuple[object, str]:
    """Fit symbolic regression: try PySR -> gplearn -> MinimalGP.

    Parameters
    ----------
    X_lagged : ndarray, shape (n_samples, n_features)
        Time-lagged input features.
    y : ndarray, shape (n_samples,)
        Target output at each timestep.
    output_idx : int
        Index of the output neuron (for logging).
    pysr_kwargs : dict, optional
        Extra kwargs for PySR if available.
    random_state : int

    Returns
    -------
    model : fitted model (PySRRegressor, SymbolicRegressor, or MinimalGP)
    backend : str
        Which backend was used: 'pysr', 'gplearn', or 'minimal_gp'.
    """
    # Epsilon-safe standardization (avoids NaN from zero-variance features)
    _eps = 1e-8
    x_mean = X_lagged.mean(axis=0)
    x_std = X_lagged.std(axis=0) + _eps
    X_scaled = (X_lagged - x_mean) / x_std

    y_mean = y.mean()
    y_std = y.std() + _eps
    y_scaled = (y - y_mean) / y_std

    # Attempt 1: PySR
    try:
        from pysr import PySRRegressor
        logger.info("  Output %d: using PySR backend", output_idx)
        default_kwargs = dict(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "exp", "log", "sqrt", "square", "abs"],
            populations=20,
            population_size=50,
            maxsize=30,
            parsimony=0.01,
            random_state=random_state,
            verbosity=0,
            progress=False,
        )
        if pysr_kwargs:
            default_kwargs.update(pysr_kwargs)
        model = PySRRegressor(**default_kwargs)
        model.fit(X_scaled, y_scaled)
        model._descartes_scaler_x = scaler_x
        model._descartes_scaler_y = scaler_y
        return model, 'pysr'
    except ImportError:
        pass

    # Attempt 2: gplearn
    try:
        from gplearn.genetic import SymbolicRegressor
        logger.info("  Output %d: using gplearn backend", output_idx)
        model = SymbolicRegressor(
            population_size=500,
            generations=30,
            stopping_criteria=1e-6,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            parsimony_coefficient=0.01,
            random_state=random_state,
            verbose=0,
        )
        model.fit(X_scaled, y_scaled)
        model._descartes_scaler_x = scaler_x
        model._descartes_scaler_y = scaler_y
        return model, 'gplearn'
    except ImportError:
        pass

    # Attempt 3: MinimalGP
    logger.info("  Output %d: using MinimalGP fallback", output_idx)
    model = MinimalGP(
        population_size=200,
        generations=50,
        max_depth=5,
        random_state=random_state,
    )
    model.fit(X_scaled, y_scaled)
    model._descartes_scaler_x = scaler_x
    model._descartes_scaler_y = scaler_y
    return model, 'minimal_gp'


# ---------------------------------------------------------------------------
# Hidden state extraction from symbolic expressions
# ---------------------------------------------------------------------------

def extract_symbolic_hidden_states(
    model,
    backend: str,
    X_lagged: np.ndarray,
) -> np.ndarray:
    """Extract 'hidden state' analog from a symbolic regression model.

    For symbolic expressions, the hidden state is the set of
    sub-expression values evaluated at each timestep.

    Parameters
    ----------
    model : fitted symbolic model
    backend : str
        'pysr', 'gplearn', or 'minimal_gp'.
    X_lagged : ndarray, shape (n_samples, n_features)

    Returns
    -------
    H : ndarray, shape (n_samples, n_sub_expressions)
        Sub-expression activations at each timestep.
    """
    scaler_x = getattr(model, '_descartes_scaler_x', None)
    if scaler_x is not None:
        X_scaled = scaler_x.transform(X_lagged)
    else:
        X_scaled = X_lagged

    if backend == 'pysr':
        return _extract_pysr_hidden(model, X_scaled)
    elif backend == 'gplearn':
        return _extract_gplearn_hidden(model, X_scaled)
    elif backend == 'minimal_gp':
        return _extract_minimal_gp_hidden(model, X_scaled)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _extract_pysr_hidden(model, X: np.ndarray) -> np.ndarray:
    """Extract sub-expression values from PySR Pareto-optimal expressions."""
    n_samples = X.shape[0]
    all_sub = []

    try:
        # PySR stores multiple equations on the Pareto front
        equations = model.equations_
        if hasattr(equations, 'iterrows'):
            for idx, row in equations.iterrows():
                try:
                    pred = model.predict(X, index=idx)
                    all_sub.append(np.nan_to_num(pred, nan=0.0))
                except Exception:
                    pass
        # If we got sub-expressions, stack them
        if all_sub:
            return np.column_stack(all_sub)
    except Exception:
        pass

    # Fallback: just use the best prediction as a single feature
    try:
        pred = model.predict(X)
        return np.nan_to_num(pred, nan=0.0).reshape(-1, 1)
    except Exception:
        return np.zeros((n_samples, 1))


def _extract_gplearn_hidden(model, X: np.ndarray) -> np.ndarray:
    """Extract sub-expression values from gplearn best program."""
    n_samples = X.shape[0]
    try:
        # gplearn stores the best program as a tree
        program = model._program
        # Execute the full program to get prediction
        pred = program.execute(X)
        # For sub-expressions, evaluate each sub-tree
        all_sub = [np.nan_to_num(pred, nan=0.0)]

        # Walk the program tree for sub-expression values
        if hasattr(program, 'program'):
            raw = program.program
            # Extract values at intermediate nodes
            for i in range(min(len(raw), 20)):
                try:
                    sub_prog = program.__class__(
                        function_set=program.function_set,
                        arities=program.arities,
                        init_depth=(1, 2),
                        init_method='full',
                        n_features=X.shape[1],
                        random_state=0,
                    )
                    sub_prog.program = raw[:i + 1]
                    sub_val = sub_prog.execute(X)
                    all_sub.append(np.nan_to_num(sub_val, nan=0.0))
                except Exception:
                    continue

        return np.column_stack(all_sub) if all_sub else pred.reshape(-1, 1)
    except Exception:
        return np.zeros((n_samples, 1))


def _extract_minimal_gp_hidden(model, X: np.ndarray) -> np.ndarray:
    """Extract sub-expression values from MinimalGP best program."""
    n_samples = X.shape[0]
    if model.best_program_ is None:
        return np.zeros((n_samples, 1))

    sub_dict = model.best_program_.extract_subexpressions(X)
    if not sub_dict:
        return np.zeros((n_samples, 1))

    # Stack all sub-expression values into a matrix
    arrays = []
    for key in sorted(sub_dict.keys()):
        val = np.nan_to_num(sub_dict[key], nan=0.0)
        val = np.clip(val, -1e6, 1e6)
        arrays.append(val)

    return np.column_stack(arrays)


def extract_random_baseline_hidden(
    n_features: int,
    X_lagged: np.ndarray,
    n_random_trees: int = 5,
    max_depth: int = 4,
    random_state: int = 0,
) -> np.ndarray:
    """Extract 'hidden states' from random (untrained) expressions.

    Generates random expression trees and evaluates their sub-expressions
    on the input data. This is the symbolic regression analog of the
    untrained LSTM baseline.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep.
    X_lagged : ndarray, shape (n_samples, n_features)
    n_random_trees : int
        Number of random trees to generate.
    max_depth : int
    random_state : int

    Returns
    -------
    H_random : ndarray, shape (n_samples, n_sub_expressions)
    """
    rng = np.random.RandomState(random_state)
    all_sub = []

    for i in range(n_random_trees):
        tree = random_expression(n_features, max_depth,
                                 np.random.RandomState(rng.randint(0, 2**31)))
        sub_dict = tree.extract_subexpressions(X_lagged)
        for key in sorted(sub_dict.keys()):
            val = np.nan_to_num(sub_dict[key], nan=0.0)
            val = np.clip(val, -1e6, 1e6)
            all_sub.append(val)

    if not all_sub:
        return np.zeros((X_lagged.shape[0], 1))

    return np.column_stack(all_sub)


# ---------------------------------------------------------------------------
# Full symbolic regression surrogate pipeline
# ---------------------------------------------------------------------------

def fit_and_extract(
    X_sequence: np.ndarray,
    Y_sequence: np.ndarray,
    window_bins: int = 50,
    lag_step: int = 5,
    random_state: int = 42,
) -> dict:
    """Full pipeline: fit symbolic surrogate and extract hidden states.

    Parameters
    ----------
    X_sequence : ndarray, shape (T, n_input)
    Y_sequence : ndarray, shape (T, n_output)
    window_bins : int
    lag_step : int
    random_state : int

    Returns
    -------
    result : dict with keys:
        'models': list of (model, backend) per output neuron
        'hidden_trained': ndarray (T', total_sub_expressions)
        'hidden_untrained': ndarray (T', total_sub_random)
        'predictions': ndarray (T', n_output)
        'test_r2': float, mean R2 on held-out data
        'backend': str
        'pareto_fronts': list of Pareto info per output
    """
    n_output = Y_sequence.shape[1] if Y_sequence.ndim > 1 else 1
    if Y_sequence.ndim == 1:
        Y_sequence = Y_sequence.reshape(-1, 1)

    # Build time-lagged features
    X_lagged = build_timelag_features(X_sequence, window_bins, lag_step)
    Y_aligned = Y_sequence[window_bins - 1:]
    n_samples = min(X_lagged.shape[0], Y_aligned.shape[0])
    X_lagged = X_lagged[:n_samples]
    Y_aligned = Y_aligned[:n_samples]

    n_features = X_lagged.shape[1]

    # Train/test split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_lagged[:split_idx], X_lagged[split_idx:]
    Y_train, Y_test = Y_aligned[:split_idx], Y_aligned[split_idx:]

    models = []
    all_hidden_trained = []
    all_predictions = []
    test_r2s = []
    pareto_fronts = []

    for out_idx in range(n_output):
        y_train = Y_train[:, out_idx]
        y_test = Y_test[:, out_idx]

        # Fit symbolic regression
        model, backend = fit_symbolic_surrogate(
            X_train, y_train, output_idx=out_idx, random_state=random_state
        )
        models.append((model, backend))

        # Extract hidden states from TRAINED model (full dataset)
        H_trained = extract_symbolic_hidden_states(model, backend, X_lagged)
        all_hidden_trained.append(H_trained)

        # Predictions on test set
        scaler_x = getattr(model, '_descartes_scaler_x', None)
        scaler_y = getattr(model, '_descartes_scaler_y', None)
        X_test_s = scaler_x.transform(X_test) if scaler_x else X_test

        if backend == 'minimal_gp':
            pred_scaled = model.predict(X_test_s)
        else:
            pred_scaled = model.predict(X_test_s)

        if scaler_y is not None:
            pred = scaler_y.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).ravel()
        else:
            pred = pred_scaled

        all_predictions.append(pred)

        # Test R2
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        test_r2s.append(float(r2))

        # Pareto front info
        if backend == 'minimal_gp' and hasattr(model, 'pareto_front_'):
            pf = [(float(mse), int(c), str(expr))
                   for mse, c, expr in model.pareto_front_[:10]]
            pareto_fronts.append(pf)
        else:
            pareto_fronts.append([])

        logger.info("  Output %d: R2=%.3f, backend=%s", out_idx, r2, backend)

    # Stack hidden states across outputs
    hidden_trained = np.hstack(all_hidden_trained) if all_hidden_trained else \
        np.zeros((n_samples, 1))

    # Random baseline hidden states
    hidden_untrained = extract_random_baseline_hidden(
        n_features, X_lagged, n_random_trees=5, max_depth=4,
        random_state=random_state + 1000,
    )

    return {
        'models': models,
        'hidden_trained': hidden_trained,
        'hidden_untrained': hidden_untrained,
        'predictions': np.column_stack(all_predictions) if all_predictions else np.zeros((n_samples - split_idx, n_output)),
        'test_r2': float(np.mean(test_r2s)),
        'backend': models[0][1] if models else 'none',
        'pareto_fronts': pareto_fronts,
        'X_lagged': X_lagged,
        'Y_aligned': Y_aligned,
        'window_bins': window_bins,
        'lag_step': lag_step,
    }
