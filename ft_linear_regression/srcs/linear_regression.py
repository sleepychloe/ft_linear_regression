from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

class LinearRegression:
        def __init__(self):
                self.learning_rate = 0.05
                self.max_iter = 1000
                self.epsilon = 1e-5
                self.theta: Vector | None = None
                self.mu: Vector | None = None
                self.sigma: Vector | None = None

        def normalize_X(self, X_raw: Matrix) -> NDArray[np.float64]:
                self.mu = X_raw.mean(axis = 0)
                self.sigma = X_raw.std(axis = 0)
                X_norm: NDArray[np.float64] = np.column_stack((np.ones(len(X_raw)), (X_raw - self.mu) / self.sigma))
                return X_norm

        def hypothesis(self, X: Matrix) -> Vector:
                if self.theta is None:
                        raise RuntimeError("Model is not fitted yet")
                return X @ self.theta

        def mse_cost(self, X: Matrix, y: Vector) -> float:
                m = len(y)
                errors = self.hypothesis(X) - y
                return (1 / (2 * m)) * np.sum(errors ** 2)

        def gradient_step(self, X: Matrix, y: Vector) -> None:
                m = len(y)
                errors = self.hypothesis(X) - y
                gradient = (1 / m) * (X.T @ errors)
                self.theta -= self.learning_rate * gradient

        def unnormalize_theta(self) -> NDArray[np.float64]:
                if self.theta is None:
                        raise RuntimeError("Model is not fitted yet")
                theta_real = self.theta.copy()
                theta_real[0] = self.theta[0] - np.sum(self.theta[1:] * self.mu / self.sigma)
                theta_real[1:] = self.theta[1:] / self.sigma
                return theta_real
