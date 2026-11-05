# Machine Learning Fundamentals

My learning repository for machine learning, built while studying Software Engineering. Each project is solved twice where it makes sense: first from scratch to understand how it works, then with a library to see how it is done in practice.

<p>
<img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"> <img src="https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> <img src="https://img.shields.io/badge/SCIKIT_LEARN-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
</p>

## Projects

| Project | What it covers | Status |
|---------|----------------|--------|
| [01 Linear Regression](01-linear-regression) | Predicting house prices from size, from scratch and with scikit-learn | Done |
| 02 Classification | Predicting categories | Planned |
| 03 Model Evaluation | Train/test splits, error metrics, overfitting | Planned |

## 01 Linear Regression

**Problem:** predict a house price (in thousands) from its size in square metres.

**What I did:**
1. [From scratch](01-linear-regression/linear_regression_from_scratch.py): NumPy only, using gradient descent to learn the slope and intercept.
2. [With scikit-learn](01-linear-regression/house_prices_sklearn.py): the same problem and the same train/test split using `LinearRegression`.

**Results on the test set:**

| Version | Slope | Intercept | MSE | R² |
|---------|-------|-----------|-----|----|
| From scratch | 1.4328 | 9.0662 | 188.76 | 0.9642 |
| scikit-learn | 1.4328 | 9.0662 | 188.76 | 0.9642 |

Both versions give the same answer, which confirms the from-scratch version is correct.

> The dataset in `data.csv` is a small sample I generated for learning. It is not real market data.

## Run It

1. Clone the repository: `git clone https://github.com/ojodavidoluwadarasimi/machine-learning-fundamentals.git`
2. Go into the folder: `cd machine-learning-fundamentals`
3. Install the libraries: `pip install numpy pandas scikit-learn`
4. Run a script: `python 01-linear-regression/linear_regression_from_scratch.py`

## Goal

Understand how and why each model works, not just call the library.
