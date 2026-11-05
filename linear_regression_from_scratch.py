"""
Project: Linear regression from scratch (NumPy only)

Problem: Predict a house price (in thousands) from its size (in square metres).

Approach:
  1. Load the data and split it into training and test sets.
  2. Scale the size values so gradient descent converges quickly.
  3. Start with slope = 0 and intercept = 0, then repeatedly nudge them in the
     direction that reduces the mean squared error (gradient descent).
  4. Convert the learned values back to the original units.
  5. Measure how well the model does on data it has not seen (test set).

Model:  price = slope * size + intercept
"""

from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data.csv"


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1]


def train_test_split(x: np.ndarray, y: np.ndarray, test_fraction: float = 0.2):
    """Split without shuffling: the last rows become the test set."""
    split = int(len(x) * (1 - test_fraction))
    return x[:split], x[split:], y[:split], y[split:]


def gradient_descent(
    x: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, steps: int = 1000
) -> tuple[float, float]:
    """Learn slope and intercept on scaled x values."""
    slope, intercept = 0.0, 0.0
    n = len(x)

    for _ in range(steps):
        predictions = slope * x + intercept
        errors = predictions - y

        # Gradients of the mean squared error
        slope_gradient = (2 / n) * np.sum(errors * x)
        intercept_gradient = (2 / n) * np.sum(errors)

        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient

    return slope, intercept


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_residual / ss_total)


def main() -> None:
    x, y = load_data(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Scale x so the numbers are small and gradient descent behaves well
    x_mean, x_std = x_train.mean(), x_train.std()
    x_train_scaled = (x_train - x_mean) / x_std

    scaled_slope, scaled_intercept = gradient_descent(x_train_scaled, y_train)

    # Convert back to the original units (square metres)
    slope = scaled_slope / x_std
    intercept = scaled_intercept - slope * x_mean

    test_predictions = slope * x_test + intercept

    print("Linear regression from scratch")
    print(f"  slope:     {slope:.4f}  (thousands per extra square metre)")
    print(f"  intercept: {intercept:.4f}")
    print(f"  test MSE:  {mean_squared_error(y_test, test_predictions):.2f}")
    print(f"  test R^2:  {r_squared(y_test, test_predictions):.4f}")


if __name__ == "__main__":
    main()
