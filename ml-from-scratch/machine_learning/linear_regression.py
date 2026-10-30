"""Linear Regression implementation from scratch.

This module provides a production-ready, educational implementation of linear
regression supporting both gradient descent and closed-form (normal equation)
solvers, with L1/L2/ElasticNet regularization, early stopping, and learning
rate scheduling.

Author: Siddhesh Sawant (ml-from-scratch)
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Solver(str, Enum):
    """Optimization strategy for fitting the model."""

    GRADIENT_DESCENT = "gradient_descent"
    NORMAL_EQUATION = "normal_equation"


class Regularization(str, Enum):
    """Type of regularization applied during fitting."""

    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    ELASTIC_NET = "elastic_net"


class LearningRateSchedule(str, Enum):
    """Learning rate schedule for gradient descent."""

    CONSTANT = "constant"
    STEP_DECAY = "step_decay"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class LinearRegression:
    """Linear regression model with multiple solvers and regularizations.

    Supports fitting via gradient descent (batch, mini-batch, or stochastic)
    or a closed-form normal equation solution. Regularization (L1, L2,
    ElasticNet) and early stopping are available for gradient descent.

    Parameters
    ----------
    solver : {Solver.GRADIENT_DESCENT, Solver.NORMAL_EQUATION} or str, default=Solver.GRADIENT_DESCENT
        Optimization strategy. ``"gradient_descent"`` uses iterative updates;
        ``"normal_equation"`` computes the closed-form solution.

        .. note::
           The normal equation with L1 or ElasticNet regularization is not
           analytically solvable. When ``solver="normal_equation"`` and
           ``regularization`` is ``"l1"`` or ``"elastic_net"``, the solver
           silently falls back to L2 regularization internally.

    learning_rate : float, default=0.01
        Step size for gradient descent. Must be positive.

    epochs : int, default=1000
        Number of passes over the data for gradient descent. Must be positive.

    batch_size : int or None, default=None
        Mini-batch size for gradient descent. ``None`` means full-batch
        gradient descent. ``1`` means stochastic gradient descent.

    regularization : {Regularization.NONE, Regularization.L1, Regularization.L2,
                      Regularization.ELASTIC_NET} or str, default=Regularization.NONE
        Type of regularization.

    lambda_ : float, default=0.0
        Regularization strength. Must be non-negative.

    l1_ratio : float, default=0.5
        Mix parameter for ElasticNet (``l1_ratio * L1 + (1 - l1_ratio) * L2``).
        Only used when ``regularization="elastic_net"``. Range: ``[0, 1]``.

    early_stopping : bool, default=False
        Whether to stop training when validation loss stops improving.

    patience : int, default=10
        Number of epochs with no improvement after which training is stopped.

    min_delta : float, default=1e-4
        Minimum change in validation loss to qualify as an improvement.

    restore_best_weights : bool, default=True
        Whether to restore model weights from the epoch with the best
        validation loss when early stopping triggers.

    lr_schedule : {LearningRateSchedule.CONSTANT, LearningRateSchedule.STEP_DECAY,
                   LearningRateSchedule.EXPONENTIAL, LearningRateSchedule.COSINE} or str, \
    default=LearningRateSchedule.CONSTANT
        Learning rate schedule. Only used for gradient descent.

    lr_decay_rate : float, default=0.9
        Decay rate for step and exponential schedules. Range: ``(0, 1]``.

    lr_decay_steps : int, default=100
        Step interval for step-decay schedule. Must be positive.

    random_state : int or None, default=None
        Seed for reproducibility of weight initialization and shuffling.

    verbose : bool, default=False
        Whether to print progress during training.

    Attributes
    ----------
    weights : NDArray[np.float64] or None
        Learned feature coefficients (shape: ``(n_features,)``).

    bias : float
        Learned intercept term.

    loss_history_ : list[float]
        Training loss at each epoch (gradient descent only).

    val_loss_history_ : list[float]
        Validation loss at each epoch if early stopping is enabled.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([3, 5, 7, 9])  # y = 2 * x + 1
    >>> model = LinearRegression(solver="normal_equation")
    >>> model.fit(X, y)
    LinearRegression(...)
    >>> model.predict(np.array([[5]]))
    array([11.])
    """

    # -- Construction ---------------------------------------------------------

    def __init__(
        self,
        solver: Union[Solver, str] = Solver.GRADIENT_DESCENT,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        regularization: Union[Regularization, str] = Regularization.NONE,
        lambda_: float = 0.0,
        l1_ratio: float = 0.5,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        lr_schedule: Union[LearningRateSchedule, str] = LearningRateSchedule.CONSTANT,
        lr_decay_rate: float = 0.9,
        lr_decay_steps: int = 100,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # Validate parameters immediately (fail-fast)
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}.")
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive or None, got {batch_size}.")

        self.solver = Solver(solver)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = Regularization(regularization)
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.lr_schedule = LearningRateSchedule(lr_schedule)
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: float = 0.0
        self.loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []
        self._n_features: int = 0

    # -- Public API -----------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            The fitted estimator (for method chaining).

        Raises
        ------
        ValueError
            If input data is invalid or parameter constraints are violated.
        """
        X, y = self._validate_input(X, y)
        self._n_features = X.shape[1]
        self._validate_params()

        # The normal equation requires L2 or no regularization.  Fall back
        # gracefully so the user gets a sensible result instead of a crash.
        if self.solver == Solver.NORMAL_EQUATION and self.regularization in (
            Regularization.L1,
            Regularization.ELASTIC_NET,
        ):
            warnings.warn(
                "L1/ElasticNet regularization is not analytically solvable with the "
                "normal equation. Solving without regularization instead.",
                UserWarning,
                stacklevel=2,
            )
            effective_reg = Regularization.NONE
        else:
            effective_reg = self.regularization

        if self.solver == Solver.NORMAL_EQUATION:
            self._fit_normal_equation(X, y, effective_reg)
        else:
            self._fit_gradient_descent(X, y, effective_reg)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict target values for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : NDArray[np.float64] of shape (n_samples,)
            Predicted target values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before calling 'predict'.")
        X = self._as_2d(X)
        return X @ self.weights + self.bias

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Coefficient of determination :math:`R^2` of the prediction.

        The best possible score is 1.0; it can be negative (because the model
        can be arbitrarily worse than a constant predictor).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            :math:`R^2` coefficient of determination.
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=float).ravel()
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    # sklearn-compatible API --------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Ignored (kept for API compatibility with scikit-learn).

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "solver": self.solver,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "regularization": self.regularization,
            "lambda_": self.lambda_,
            "l1_ratio": self.l1_ratio,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best_weights": self.restore_best_weights,
            "lr_schedule": self.lr_schedule,
            "lr_decay_rate": self.lr_decay_rate,
            "lr_decay_steps": self.lr_decay_steps,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "LinearRegression":
        """Set parameters for this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters to update.

        Returns
        -------
        self : LinearRegression
            The estimator instance.
        """
        valid_keys = {
            "solver", "learning_rate", "epochs", "batch_size",
            "regularization", "lambda_", "l1_ratio",
            "early_stopping", "patience", "min_delta", "restore_best_weights",
            "lr_schedule", "lr_decay_rate", "lr_decay_steps",
            "random_state", "verbose",
        }
        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter '{key}'.")
            setattr(self, key, value)
        return self

    # -- Educational visualization -------------------------------------------

    def plot_loss_history(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the training loss curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on. If ``None``, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.loss_history_, label="Training Loss")
        if self.val_loss_history_:
            ax.plot(self.val_loss_history_, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.set_title("Loss History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_coefficients(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot learned coefficients.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        feature_names = [f"x{i}" for i in range(self._n_features)]
        ax.bar(feature_names, self.weights)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Coefficient")
        ax.set_title("Learned Coefficients")
        ax.grid(True, alpha=0.3, axis="y")
        return ax

    def plot_predictions(
        self, X: ArrayLike, y: ArrayLike, ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot actual vs. predicted values (1-D only).

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Single-feature input used for plotting.
        y : array-like of shape (n_samples,)
            True target values.
        ax : matplotlib.axes.Axes or None, default=None
            Axes to draw on.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        X_1d = np.asarray(X, dtype=float).ravel()
        if self._n_features != 1:
            warnings.warn(
                "plot_predictions only supports single-feature models.",
                UserWarning,
            )

        order = np.argsort(X_1d)
        y_arr = np.asarray(y, dtype=float).ravel()
        preds = self.predict(X_1d)

        ax.scatter(X_1d, y_arr, label="Actual", alpha=0.6)
        ax.plot(X_1d[order], preds[order], color="red", label="Predicted", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"R² = {self.score(X_1d, y_arr):.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # -- Solver implementations ----------------------------------------------

    def _fit_gradient_descent(self, X: NDArray[np.float64], y: NDArray[np.float64],
                              effective_reg: Regularization) -> None:
        """Fit using mini-batch / batch / SGD gradient descent."""
        n_samples = X.shape[0]

        # Weight initialization
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.random(self._n_features)
        self.bias = 0.0
        self.loss_history_ = []
        self.val_loss_history_ = []

        # Early-stopping bookkeeping
        best_weights = None
        best_bias = None
        best_val_loss = float("inf")
        best_epoch = 0
        no_improve_count = 0

        # Batch size resolution
        if self.batch_size is None:
            actual_batch = n_samples
        else:
            actual_batch = min(self.batch_size, n_samples)
        n_batches = max(1, (n_samples + actual_batch - 1) // actual_batch)

        for epoch in range(self.epochs):
            # Shuffle once per epoch for mini-batch SGD
            perm = rng.permutation(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            epoch_loss_sum = 0.0

            for i in range(n_batches):
                start = i * actual_batch
                end = min(start + actual_batch, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                y_pred = X_batch @ self.weights + self.bias
                error = y_pred - y_batch
                n_b = len(y_batch)

                # Data gradient (MSE)
                dw = (2.0 / n_b) * (X_batch.T @ error)
                db = (2.0 / n_b) * np.sum(error)

                # Regularization gradient (L1 / L2 / ElasticNet)
                # sign(0) = 0 → subgradient at origin for L1
                if effective_reg == Regularization.L1:
                    dw += (self.lambda_ / n_samples) * np.sign(self.weights)
                elif effective_reg == Regularization.L2:
                    dw += (2.0 * self.lambda_ / n_samples) * self.weights
                elif effective_reg == Regularization.ELASTIC_NET:
                    l1_term = (self.lambda_ / n_samples) * np.sign(self.weights)
                    l2_term = (2.0 * self.lambda_ * (1 - self.l1_ratio) / n_samples) * self.weights
                    dw += self.l1_ratio * l1_term + l2_term

                # Parameter update
                lr = self._compute_lr(epoch)
                self.weights -= lr * dw
                self.bias -= lr * db

                batch_loss = float(np.mean(error ** 2))
                epoch_loss_sum += batch_loss * n_b

            epoch_loss = epoch_loss_sum / n_samples
            self.loss_history_.append(epoch_loss)

            # Early stopping
            if self.early_stopping:
                val_loss = self._compute_val_loss(X, y, effective_reg)
                self.val_loss_history_.append(val_loss)
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    best_epoch = epoch
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if self.verbose:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs} | "
                        f"loss={epoch_loss:.6f} | val_loss={val_loss:.6f} | "
                        f"lr={lr:.6f}"
                    )
                if no_improve_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}. Best at {best_epoch + 1}.")
                    if self.restore_best_weights and best_weights is not None:
                        self.weights = best_weights
                        self.bias = best_bias
                    break

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} | loss={epoch_loss:.6f} | lr={lr:.6f}")

    def _fit_normal_equation(
        self, X: NDArray[np.float64], y: NDArray[np.float64],
        effective_reg: Regularization,
    ) -> None:
        """Fit using the normal equation (closed form).

        For L2 regularization:  θ = (XᵀX + λI)⁻¹ Xᵀy

        For no regularization:  θ = (XᵀX)⁻¹ Xᵀy

        Uses ``np.linalg.pinv`` to handle singular or near-singular matrices.
        """
        n_samples, n_features = X.shape

        # Augment X with a bias column of ones
        X_aug = np.hstack([X, np.ones((n_samples, 1))])  # shape (n, n_features+1)

        # Regularization matrix (don't regularize the bias term)
        reg_matrix = np.zeros((n_features + 1, n_features + 1))
        if effective_reg == Regularization.L2:
            # Scale by n_samples so it matches the GD convention
            reg_matrix[:n_features, :n_features] = (2.0 * self.lambda_ / n_samples) * np.eye(n_features)

        # Normal equation:  θ = (XᵀX + R)⁻¹ Xᵀy
        XTX = X_aug.T @ X_aug + reg_matrix
        XTy = X_aug.T @ y
        theta = np.linalg.pinv(XTX) @ XTy  # pinv handles singular matrices

        self.weights = theta[:-1]
        self.bias = float(theta[-1])
        self.loss_history_ = [float(self._compute_loss(y, X_aug @ theta))]
        self.val_loss_history_ = []

    # -- Helpers --------------------------------------------------------------

    def _compute_lr(self, epoch: int) -> float:
        """Compute the learning rate for *epoch* under the configured schedule."""
        if self.lr_schedule == LearningRateSchedule.CONSTANT:
            return self.learning_rate
        if self.lr_schedule == LearningRateSchedule.STEP_DECAY:
            return self.learning_rate * (self.lr_decay_rate ** (epoch // self.lr_decay_steps))
        if self.lr_schedule == LearningRateSchedule.EXPONENTIAL:
            return self.learning_rate * np.exp(-self.lr_decay_rate * epoch)
        if self.lr_schedule == LearningRateSchedule.COSINE:
            return (
                self.learning_rate
                * 0.5
                * (1 + np.cos(np.pi * epoch / self.epochs))
            )
        return self.learning_rate  # Should never happen

    def _compute_val_loss(
        self, X: NDArray[np.float64], y: NDArray[np.float64],
        effective_reg: Regularization,
    ) -> float:
        """Compute validation loss including regularization penalty."""
        y_pred = X @ self.weights + self.bias
        mse = float(np.mean((y_pred - y) ** 2))
        reg_penalty = self._reg_penalty()
        return mse + reg_penalty

    def _reg_penalty(self) -> float:
        """Compute the current regularization penalty term."""
        if self.regularization == Regularization.NONE or self.weights is None:
            return 0.0
        w = self.weights
        if self.regularization == Regularization.L1:
            return self.lambda_ * float(np.sum(np.abs(w)))
        if self.regularization == Regularization.L2:
            return self.lambda_ * float(np.sum(w ** 2))
        if self.regularization == Regularization.ELASTIC_NET:
            l1 = self.lambda_ * float(np.sum(np.abs(w)))
            l2 = self.lambda_ * float(np.sum(w ** 2))
            return self.l1_ratio * l1 + (1 - self.l1_ratio) * l2
        return 0.0

    def _compute_loss(self, y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """Compute mean squared error."""
        return float(np.mean((y_pred - y) ** 2))

    def _validate_input(self, X: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate and convert inputs to numpy arrays.

        Returns
        -------
        X : NDArray[np.float64] of shape (n_samples, n_features)
        y : NDArray[np.float64] of shape (n_samples,)
        """
        X = self._as_2d(X)
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have the same number of samples: "
                f"got X.shape={X.shape[0]}, y.shape={y.shape[0]}."
            )
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values (no NaN or inf).")

        return X, y

    def _validate_params(self) -> None:
        """Raise ``ValueError`` if constructor parameters are invalid."""
        # Core validation done in __init__ for fail-fast behavior
        if self.early_stopping and self.solver == Solver.NORMAL_EQUATION:
            warnings.warn(
                "Early stopping is not applicable with the normal equation solver; "
                "it will be ignored.",
                UserWarning,
            )

    # -- Shared utilities (static) -------------------------------------------

    @staticmethod
    def _as_2d(X: ArrayLike) -> NDArray[np.float64]:
        """Convert input to a 2-D float array."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1-D or 2-D.")
        return X

    def __repr__(self) -> str:
        return (
            f"LinearRegression(solver={self.solver.value!r}, "
            f"learning_rate={self.learning_rate}, "
            f"epochs={self.epochs}, "
            f"regularization={self.regularization.value!r}, "
            f"lambda_={self.lambda_})"
        )


# ---------------------------------------------------------------------------
# Standalone plotting helpers (backward-compatible convenience functions)
# ---------------------------------------------------------------------------

def plot_fit(
    ax: plt.Axes,
    x: ArrayLike,
    y: ArrayLike,
    predictions: ArrayLike,
    xlabel: str = "x",
    ylabel: str = "y",
) -> None:
    """Plot a scatter of actual vs. predicted data on the given axes."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    pred_arr = np.asarray(predictions, dtype=float).ravel()
    order = np.argsort(x_arr)
    ax.scatter(x_arr, y_arr, label=f"Actual {ylabel}")
    ax.plot(x_arr[order], pred_arr[order], color="red", label=f"Predicted {ylabel}")
    ax.set(title=f"{ylabel} vs {xlabel}", xlabel=xlabel, ylabel=ylabel)
    ax.legend()


def plot_loss(ax: plt.Axes, loss_history: list[float]) -> None:
    """Plot the loss history on the given axes."""
    ax.plot(loss_history)
    ax.set(title="Loss vs Epochs", xlabel="Epoch", ylabel="Mean Squared Error")


def plot_results(
    x: ArrayLike,
    y: ArrayLike,
    predictions: ArrayLike,
    loss_history: list[float],
    xlabel: str = "Years of Experience",
    ylabel: str = "Salary",
    save_path: Optional[str] = None,
) -> None:
    """Create a side-by-side plot of data fit and loss history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_fit(axes[0], np.asarray(x), np.asarray(y), np.asarray(predictions), xlabel, ylabel)
    plot_loss(axes[1], loss_history)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic data: salary = 30 + 9 * experience + noise
    rng = np.random.default_rng(42)
    experience = rng.uniform(0, 10, 100)
    salary = 30 + 9 * experience + rng.normal(0, 5, 100)  # in $1000s

    # --- Gradient descent with early stopping ---
    gd_model = LinearRegression(
        solver=Solver.GRADIENT_DESCENT,
        learning_rate=0.01,
        epochs=5000,
        batch_size=None,  # full-batch GD
        early_stopping=True,
        patience=50,
        lr_schedule=LearningRateSchedule.COSINE,
        random_state=42,
        verbose=False,
    ).fit(experience, salary)

    g_preds = gd_model.predict(experience)
    print(
        f"[GD]   weight={gd_model.weights[0]:.3f}  bias={gd_model.bias:.3f}  "
        f"final loss={gd_model.loss_history_[-1]:.3f}  R²={gd_model.score(experience, salary):.4f}"
    )

    # --- Normal equation ---
    ne_model = LinearRegression(
        solver=Solver.NORMAL_EQUATION,
        regularization=Regularization.L2,
        lambda_=0.01,
    ).fit(experience, salary)

    ne_preds = ne_model.predict(experience)
    print(
        f"[NE]   weight={ne_model.weights[0]:.3f}  bias={ne_model.bias:.3f}  "
        f"R²={ne_model.score(experience, salary):.4f}"
    )

    # --- Visualize ---
    plot_results(
        experience, salary, g_preds, gd_model.loss_history_,
        xlabel="Years of Experience", ylabel="Salary",
    )
