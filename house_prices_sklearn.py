"""
Project: Linear regression with scikit-learn

Problem: Predict a house price (in thousands) from its size (in square metres).
This is the same problem as linear_regression_from_scratch.py, solved with
scikit-learn so the two results can be compared.

Approach:
  1. Load the data with Pandas.
  2. Split into training and test sets (same split as the scratch version:
     no shuffling, last 20 percent of rows for testing).
  3. Fit a LinearRegression model.
  4. Measure error and R^2 on the test set.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[["size_sqm"]]
    y = df["price_thousands"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Linear regression with scikit-learn")
    print(f"  slope:     {model.coef_[0]:.4f}  (thousands per extra square metre)")
    print(f"  intercept: {model.intercept_:.4f}")
    print(f"  test MSE:  {mean_squared_error(y_test, predictions):.2f}")
    print(f"  test R^2:  {r2_score(y_test, predictions):.4f}")


if __name__ == "__main__":
    main()
