import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b) where w has shape (D,) and b is a float.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    N, D = X.shape
    w = np.zeros(D, dtype=float)
    b = 0.0

    for _ in range(int(steps)):
        z = X @ w + b                 # (N,)
        p = _sigmoid(z)               # (N,)

        # Gradients of average BCE loss
        err = p - y                   # (N,)
        dw = (X.T @ err) / N          # (D,)
        db = np.sum(err) / N          # scalar

        # Parameter update
        w -= lr * dw
        b -= lr * db

    return w, float(b)
