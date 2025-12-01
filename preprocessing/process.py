import numpy as np
from scipy.signal import detrend as scipy_detrend
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

# ---- MSC----
def msc(X):
        mean_spectrum = np.mean(X, axis=0)
        corrected_X = np.zeros_like(X)
        for i in range(X.shape[0]):
            p = np.polyfit(mean_spectrum, X[i, :], 1)
            corrected_X[i, :] = (X[i, :] - p[1]) / p[0]
        return corrected_X

# ---- SNV----
def snv(X):
    return (X - np.mean(X, axis=1)[:, np.newaxis]) / np.std(X, axis=1)[:, np.newaxis]

# ---- Detrend----
def detrend(X, type='linear',axis=1):
    return scipy_detrend(X, axis=axis, type=type)

# ---- Derivative ----
def derivative(X):
        derivative_X = np.zeros_like(X)
        for i in range(X.shape[0]):
            derivative_X[i, 0] = X[i, 1] - X[i, 0] 
            for j in range(1, X.shape[1] - 1):
                derivative_X[i, j] = (X[i, j + 1] - X[i, j - 1]) / 2  
            derivative_X[i, -1] = X[i, -1] - X[i, -2]  
        return derivative_X


# ---- Savitzky-Golay----
def savitzky_golay(X, window_length=11, polyorder=2, deriv=2, axis=1):
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=axis)

# ---- airPLS----
def _whittaker_1d(x, w, lam, differences=1):
    x = np.asarray(x, dtype=float).ravel()
    m = x.size
    E = eye(m, format='csc')
    for _ in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + lam * E.T * E)
    B = csc_matrix(W * x.reshape(-1, 1))
    z = spsolve(A, B)
    return np.asarray(z).ravel()

def airPLS(X, lam=100, porder=1, itermax=15):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    X_corr = np.empty_like(X)
    for i in range(n_samples):
        x = X[i].ravel()
        w = np.ones(x.size)
        z = np.zeros_like(x)
        for it in range(1, itermax + 1):
            z = _whittaker_1d(x, w, lam, porder)
            d = x - z
            neg = d < 0
            dssn = np.abs(d[neg].sum())
            if dssn == 0:
                break
            w[~neg] = 0
            w[neg] = np.exp(it * np.abs(d[neg]) / dssn)
            w[0] = w[-1] = np.exp(it * (d[neg].max()) / dssn)
        X_corr[i, :] = x - z
    return X_corr

