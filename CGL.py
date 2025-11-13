import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing.process import derivative
from preprocessing.feature import rfe


# Load the data
train_path = "D:/A/CSU/CGL/CGL_nir_cal.xlsx"
test_path = "D:/A/CSU/CGL/CGL_nir_test.xlsx"

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train.iloc[:, :117].values
y_train = df_train.iloc[:, 120].values.ravel() #Water content
X_test = df_test.iloc[:, :117].values
y_test = df_test.iloc[:, 120].values.ravel()

# Spectral preprocessing
X_train_de = derivative(X_train)
X_test_de = derivative(X_test)

# Feature Selection
X_train_rfe, X_test_rfe = rfe(X_train_de, y_train, X_test_de, n_features_to_select=90) #n_features_to_select can be adjusted

# TabPFN
tabpfn_regressor = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42)
tabpfn_regressor.fit(X_train_rfe, y_train)
y_train_pred_tabpfn = tabpfn_regressor.predict(X_train_rfe)
y_test_pred_tabpfn = tabpfn_regressor.predict(X_test_rfe)

# PLSR
param_grid = {'n_components': [5, 10, 15, 20]}
pls = PLSRegression()
grid = GridSearchCV(pls, param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train_rfe, y_train)
best_pls = grid.best_estimator_
y_train_pred_plsr = best_pls.predict(X_train_rfe).ravel()
y_test_pred_plsr = best_pls.predict(X_test_rfe).ravel()


# Evaluate performance
def calculate_sep(y_true, y_pred):
    n = len(y_true)
    if n <= 1:
        return 0.0
    sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))
    return sep

def evaluate_performance(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    sep = calculate_sep(y_true, y_pred)
    print(f"\n--- {name} metric ---")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"SEP: {sep:.4f}")

evaluate_performance(y_train, y_train_pred_tabpfn, "TabPFN-train")
evaluate_performance(y_test, y_test_pred_tabpfn, "TabPFN-test")
evaluate_performance(y_train, y_train_pred_plsr, "PLSR-train")
evaluate_performance(y_test, y_test_pred_plsr, "PLSR-test")
