import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self, lr=0.01, epochs=1000, random_state=None):
        self.lr = lr
        self.epochs = epochs
        
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        
    def _init_params(self, n_features):
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.random(n_features)
        self.bias = 0.0
        self.loss_history = []
        
    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        return X
        
    def predict(self, X):
        return self._as_2d(X) @ self.weights + self.bias
    
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)
    
    def compute_gradients(self, X, y):
        
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        error = y_pred - y
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        
        X, y = self._as_2d(X), np.asarray(y, dtype=float).ravel()
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X cannot be empty.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values.")
        
        self._init_params(X.shape[1])
        
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            loss = self.loss(y_pred, y)
            self.loss_history.append(loss)
            
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self

def plot_fit(ax, x, y, predictions, xlabel="x", ylabel="y"):
    order = np.argsort(x)
    ax.scatter(x, y, label=f"Actual {ylabel}")
    ax.plot(x[order], predictions[order], color="red", label=f"Predicted {ylabel}")
    ax.set(title=f"{ylabel} vs {xlabel}", xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    
def plot_loss(ax, loss_history):
    ax.plot(loss_history)
    ax.set(title="Loss vs Epochs", xlabel="Epoch", ylabel="Mean Squared Error")

def plot_results(x, y, predictions, loss_history,
                 xlabel="Years of Experience", ylabel="Salary", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    plot_fit(axes[0], np.asarray(x), np.asarray(y), np.asarray(predictions), xlabel, ylabel)
    plot_loss(axes[1], loss_history)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

if __name__ == '__main__':
    
    print("hi")
    
    rng = np.random.default_rng(0)
    experience = rng.uniform(0, 10, 100)
    salary = 30 + 9 * experience + rng.normal(0, 5, 100)   # in $1000s
    
    model = LinearRegression(lr=0.01, epochs=2000, random_state=0).fit(experience, salary)
    print(f"weight={model.weights[0]:.3f}  bias={model.bias:.3f}  final loss={model.loss_history[-1]:.3f}")
    
    # plot_results(experience, salary, model.predict(experience), model.loss_history,
    #             save_path="results.png")
    
    plot_results(experience, salary, model.predict(experience), model.loss_history)
    

    
    
