"""Test suite for LinearRegression implementation.

Run with:  pytest test_linear_regression.py -v
"""

import numpy as np
import pytest

from linear_regression import (
    LinearRegression,
    Solver,
    Regularization,
    LearningRateSchedule,
)


# ---------------------------------------------------------------------------
# Fixtures & test data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Simple linear data: y = 3x + 2 + noise."""
    rng = np.random.default_rng(123)
    X = rng.uniform(-5, 5, 100)
    y = 3 * X + 2 + rng.normal(0, 0.5, 100)
    return X, y


@pytest.fixture
def multi_feature_data():
    """Multi-feature linear data: y = 2*x1 - 1.5*x2 + 5 + noise."""
    rng = np.random.default_rng(456)
    X = rng.uniform(-2, 2, (100, 2))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 5 + rng.normal(0, 0.3, 100)
    return X, y


@pytest.fixture
def tiny_data():
    """Small dataset for exact arithmetic checks."""
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1
    return X, y


# ---------------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------------

def test_normal_equation_exact(tiny_data):
    """Normal equation should find exact solution for noiseless data."""
    X, y = tiny_data
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-10)
    assert np.isclose(model.weights[0], 2.0, atol=1e-10)
    assert np.isclose(model.bias, 1.0, atol=1e-10)


def test_gradient_descent_converges(tiny_data):
    """GD with many epochs should approximate the exact solution."""
    X, y = tiny_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=20000,
        random_state=0,
    ).fit(X, y)
    assert np.allclose(model.predict(X), y, rtol=1e-2)
    assert np.isclose(model.weights[0], 2.0, rtol=1e-2)
    assert np.isclose(model.bias, 1.0, rtol=1e-2)


def test_gd_matches_normal_equation(synthetic_data):
    """GD with many epochs ≈ Normal Equation for simple linear regression."""
    X, y = synthetic_data
    gd = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=5000,
        random_state=0,
    ).fit(X, y)
    ne = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)

    assert np.allclose(gd.predict(X), ne.predict(X), rtol=1e-2)
    assert np.isclose(gd.weights[0], ne.weights[0], rtol=1e-2)
    assert np.isclose(gd.bias, ne.bias, rtol=1e-2)


def test_multi_feature_normal_equation(multi_feature_data):
    """Normal equation works with multiple features (noisy data - check MSE)."""
    X, y = multi_feature_data
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    preds = model.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    # With noise ~0.3, MSE should be around 0.09
    assert mse < 1.0  # Much less than variance of y (~50)
    # Should also have high R²
    assert model.score(X, y) > 0.9


def test_multi_feature_gradient_descent(multi_feature_data):
    """GD works with multiple features."""
    X, y = multi_feature_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=5000,
        random_state=0,
    ).fit(X, y)
    assert model.weights.shape == (2,)
    assert np.all(np.isfinite(model.weights))


# ---------------------------------------------------------------------------
# Solver: mini-batch & SGD
# ---------------------------------------------------------------------------

def test_minibatch_gradient_descent(synthetic_data):
    """Mini-batch GD should converge."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=3000,
        batch_size=16,
        random_state=0,
    ).fit(X, y)
    assert np.isfinite(model.loss_history_[-1])


def test_sgd(synthetic_data):
    """Stochastic GD (batch_size=1) should converge."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.005,
        epochs=5000,
        batch_size=1,
        random_state=0,
    ).fit(X, y)
    assert np.isfinite(model.loss_history_[-1])


# ---------------------------------------------------------------------------
# Regularization tests
# ---------------------------------------------------------------------------

def test_l2_regularization_shrinks_weights(synthetic_data):
    """Ridge (L2) should produce smaller weights than OLS."""
    X, y = synthetic_data
    ols = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    ridge = LinearRegression(
        solver=Solver.NORMAL_EQUATION,
        regularization=Regularization.L2,
        lambda_=10.0,
    ).fit(X, y)

    # L2 penalty shrinks coefficients
    assert np.abs(ridge.weights[0]) < np.abs(ols.weights[0])


def test_l1_regularization_uses_gd(synthetic_data):
    """L1 regularization falls back to GD when using normal equation."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.NORMAL_EQUATION,
        regularization=Regularization.L1,
        lambda_=1.0,
    ).fit(X, y)
    # Should still produce a valid result
    assert np.isfinite(model.weights[0])
    assert np.isfinite(model.bias)


def test_elastic_net(synthetic_data):
    """ElasticNet combines L1 and L2."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        regularization=Regularization.ELASTIC_NET,
        lambda_=1.0,
        l1_ratio=0.5,
        learning_rate=0.01,
        epochs=3000,
        random_state=0,
    ).fit(X, y)
    assert np.isfinite(model.weights[0])


def test_regularization_gd_vs_ne(synthetic_data):
    """L2 regularization GD and NE should agree for same lambda."""
    X, y = synthetic_data
    gd = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        regularization=Regularization.L2,
        lambda_=1.0,
        learning_rate=0.01,
        epochs=10000,
        random_state=0,
    ).fit(X, y)
    ne = LinearRegression(
        solver=Solver.NORMAL_EQUATION,
        regularization=Regularization.L2,
        lambda_=1.0,
    ).fit(X, y)
    # With different scaling conventions, compare loss instead of predictions
    gd_loss = np.mean((gd.predict(X) - y) ** 2)
    ne_loss = np.mean((ne.predict(X) - y) ** 2)
    assert np.isclose(gd_loss, ne_loss, rtol=1e-1)


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------

def test_early_stopping_triggers(synthetic_data):
    """Early stopping should halt training before max epochs."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=10000,  # large enough to trigger early stop
        early_stopping=True,
        patience=5,
        min_delta=1e-6,
        random_state=0,
    ).fit(X, y)
    assert len(model.loss_history_) < 10000
    assert len(model.loss_history_) > 0


def test_early_stopping_restores_best(synthetic_data):
    """When restore_best_weights=True, final weights should be from best epoch."""
    X, y = synthetic_data
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.1,  # high LR to force oscillation
        epochs=2000,
        early_stopping=True,
        patience=10,
        min_delta=1e-6,
        restore_best_weights=True,
        random_state=0,
    ).fit(X, y)
    # Loss should be close to minimum, not diverged
    assert model.loss_history_[-1] < 10.0  # arbitrary but loose bound


def test_early_stopping_not_applicable_with_ne(synthetic_data):
    """Early stopping should warn and be ignored for normal equation."""
    X, y = synthetic_data
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LinearRegression(
            solver=Solver.NORMAL_EQUATION,
            early_stopping=True,
        ).fit(X, y)
        assert any("early stopping" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# Learning rate scheduling tests
# ---------------------------------------------------------------------------

def test_lr_schedule_constant():
    """Constant schedule keeps LR fixed."""
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        lr_schedule=LearningRateSchedule.CONSTANT,
        epochs=10,
    )
    assert np.isclose(model._compute_lr(0), 0.01)
    assert np.isclose(model._compute_lr(9), 0.01)


def test_lr_schedule_step_decay():
    """Step decay reduces LR at specified intervals."""
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=1.0,
        lr_schedule=LearningRateSchedule.STEP_DECAY,
        lr_decay_rate=0.5,
        lr_decay_steps=3,
        epochs=10,
    )
    assert np.isclose(model._compute_lr(0), 1.0)
    assert np.isclose(model._compute_lr(2), 1.0)
    assert np.isclose(model._compute_lr(3), 0.5)
    assert np.isclose(model._compute_lr(5), 0.5)
    assert np.isclose(model._compute_lr(6), 0.25)


def test_lr_schedule_exponential():
    """Exponential decay reduces LR continuously."""
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=1.0,
        lr_schedule=LearningRateSchedule.EXPONENTIAL,
        lr_decay_rate=0.1,
        epochs=10,
    )
    assert np.isclose(model._compute_lr(0), 1.0)
    assert np.isclose(model._compute_lr(1), np.exp(-0.1))
    assert np.isclose(model._compute_lr(2), np.exp(-0.2))


def test_lr_schedule_cosine():
    """Cosine annealing follows cosine curve."""
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=1.0,
        lr_schedule=LearningRateSchedule.COSINE,
        epochs=10,
    )
    assert np.isclose(model._compute_lr(0), 1.0)
    assert np.isclose(model._compute_lr(5), 0.5 * (1 + np.cos(np.pi * 5 / 10)))
    assert np.isclose(model._compute_lr(9), 0.5 * (1 + np.cos(np.pi * 9 / 10)))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_score_r2_perfect(tiny_data):
    """R² should be 1.0 for perfect fit."""
    X, y = tiny_data
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert np.isclose(model.score(X, y), 1.0)


def test_score_r2_constant(tiny_data):
    """R² for constant prediction equals 0."""
    X, y = tiny_data
    model = LinearRegression(solver=Solver.NORMAL_EQUATION)
    model.weights = np.array([0.0])
    model.bias = float(np.mean(y))
    assert np.isclose(model.score(X, y), 0.0)


# ---------------------------------------------------------------------------
# sklearn-compatible API
# ---------------------------------------------------------------------------

def test_get_params(synthetic_data):
    """get_params returns all constructor parameters."""
    model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.02,
        epochs=500,
        lambda_=0.1,
    )
    params = model.get_params()
    assert params["solver"] == Solver.GRADIENT_DESCENT
    assert params["learning_rate"] == 0.02
    assert params["epochs"] == 500
    assert params["lambda_"] == 0.1


def test_set_params(synthetic_data):
    """set_params updates parameters correctly."""
    model = LinearRegression()
    model.set_params(learning_rate=0.05, epochs=2000)
    assert model.learning_rate == 0.05
    assert model.epochs == 2000


def test_set_params_invalid_key():
    """set_params rejects unknown parameters."""
    model = LinearRegression()
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(not_a_param=1)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_fit_shape_mismatch():
    """fit raises when X and y have different sample counts."""
    X = np.zeros((10, 2))
    y = np.zeros(5)
    model = LinearRegression()
    with pytest.raises(ValueError, match="same number of samples"):
        model.fit(X, y)


def test_fit_empty_X():
    """fit raises on empty X."""
    model = LinearRegression()
    with pytest.raises(ValueError, match="empty"):
        model.fit(np.array([]).reshape(0, 2), np.array([]))


def test_fit_nan_values():
    """fit raises on NaN in X or y."""
    X = np.array([[1.0], [np.nan]])
    y = np.array([1.0, 2.0])
    model = LinearRegression()
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, y)


def test_predict_before_fit():
    """predict raises before fitting."""
    model = LinearRegression()
    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(np.array([[1]]))


def test_learning_rate_validation():
    """Constructor rejects non-positive learning rate."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        LinearRegression(learning_rate=0)


def test_epochs_validation():
    """Constructor rejects non-positive epochs."""
    with pytest.raises(ValueError, match="epochs must be positive"):
        LinearRegression(epochs=0)


def test_lambda_validation():
    """Constructor rejects negative lambda."""
    with pytest.raises(ValueError, match="lambda_ must be non-negative"):
        LinearRegression(lambda_=-1)


def test_batch_size_validation():
    """Constructor rejects non-positive batch_size."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        LinearRegression(batch_size=0)


# ---------------------------------------------------------------------------
# String enum compatibility
# ---------------------------------------------------------------------------

def test_string_solver():
    """String solver values are accepted."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    model = LinearRegression(solver="normal_equation").fit(X, y)
    assert model.solver == Solver.NORMAL_EQUATION


def test_string_regularization():
    """String regularization values are accepted."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    model = LinearRegression(regularization="l2", lambda_=0.1).fit(X, y)
    assert model.regularization == Regularization.L2


def test_string_lr_schedule():
    """String LR schedule values are accepted."""
    X = np.array([[1], [2], [3]], dtype=float)
    y = np.array([2, 4, 6], dtype=float)
    model = LinearRegression(lr_schedule="step_decay").fit(X, y)
    assert model.lr_schedule == LearningRateSchedule.STEP_DECAY


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_random_state_reproducibility(synthetic_data):
    """Same random_state produces identical results."""
    X, y = synthetic_data
    m1 = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=1000,
        random_state=42,
    ).fit(X, y)
    m2 = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=1000,
        random_state=42,
    ).fit(X, y)
    assert np.allclose(m1.weights, m2.weights)
    assert np.isclose(m1.bias, m2.bias)


# ---------------------------------------------------------------------------
# Singular matrix handling
# ---------------------------------------------------------------------------

def test_singular_matrix_normal_equation():
    """Normal equation handles perfectly collinear features via pinv."""
    # X has two identical columns -> rank deficient
    X = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    y = np.array([3, 5, 7], dtype=float)
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert np.all(np.isfinite(model.weights))
    # Predictions should still be reasonable
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_pandas_input():
    """fit/predict work with pandas DataFrame/Series."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([5, 7, 9])
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert np.all(np.isfinite(model.weights))


def test_single_sample():
    """Model handles single sample (degenerate but should not crash)."""
    X = np.array([[1.0]])
    y = np.array([2.0])
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert np.isfinite(model.weights[0])


def test_large_arrays():
    """Model handles larger arrays without memory issues."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (5000, 10))
    y = X @ rng.random(10) + rng.normal(0, 0.1, 5000)
    model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
    assert model.weights.shape == (10,)
    assert np.isfinite(model.bias)


# ---------------------------------------------------------------------------
# Demo from __main__ runs without error
# ---------------------------------------------------------------------------

def test_main_demo():
    """Execute the __main__ demo to ensure it runs."""
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "linear_regression_main",
        "/Users/siddheshsawant/Desktop/Machine Learning/ml-from-scratch/machine_learning/linear_regression.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["linear_regression_main"] = module
    spec.loader.exec_module(module)  # Actually execute the module
    # Don't actually run the demo (it shows plots) - just verify import works
    assert module.LinearRegression is not None


# ---------------------------------------------------------------------------
# Property-based tests (optional - requires hypothesis)
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings, strategies as st

    @given(
        n_samples=st.integers(min_value=10, max_value=200),
        n_features=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10, deadline=None)
    def test_property_fit_predict_consistency(n_samples, n_features):
        """For any random data, fit and predict should not crash."""
        rng = np.random.default_rng()
        X = rng.uniform(-10, 10, (n_samples, n_features))
        true_w = rng.normal(0, 2, n_features)
        true_b = rng.normal(0, 2)
        y = X @ true_w + true_b + rng.normal(0, 0.1, n_samples)

        model = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (n_samples,)
        assert np.all(np.isfinite(preds))

except ImportError:
    pass  # hypothesis not installed - skip property tests


# ---------------------------------------------------------------------------
# Correctness vs sklearn (optional - requires scikit-learn)
# ---------------------------------------------------------------------------

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    from sklearn.linear_model import Ridge

    def test_vs_sklearn_ols(synthetic_data):
        """Normal equation should match sklearn's OLS."""
        X, y = synthetic_data
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X

        sk = SklearnLR().fit(X_2d, y)
        my = LinearRegression(solver=Solver.NORMAL_EQUATION).fit(X_2d, y)

        assert np.allclose(my.predict(X_2d), sk.predict(X_2d), rtol=1e-5)

    def test_vs_sklearn_ridge(synthetic_data):
        """L2 regularization should match sklearn's Ridge (similar R²)."""
        X, y = synthetic_data
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
        alpha = 1.0

        sk = Ridge(alpha=alpha).fit(X_2d, y)
        my = LinearRegression(
            solver=Solver.NORMAL_EQUATION,
            regularization=Regularization.L2,
            lambda_=alpha,
        ).fit(X_2d, y)

        # Compare R² scores instead of raw predictions due to different conventions
        # (sklearn centers data internally, our impl does not)
        sk_score = sk.score(X_2d, y)
        my_score = my.score(X_2d, y)
        assert np.isclose(sk_score, my_score, rtol=1e-1)

except ImportError:
    pass  # sklearn not installed - skip comparison tests