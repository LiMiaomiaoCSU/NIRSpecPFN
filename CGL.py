import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def spectral_first_order_derivative(X):
    """光谱一阶微分处理"""
    derivative_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        derivative_X[i, 0] = X[i, 1] - X[i, 0]  # 前向差分
        for j in range(1, X.shape[1] - 1):
            derivative_X[i, j] = (X[i, j + 1] - X[i, j - 1]) / 2  # 中心差分
        derivative_X[i, -1] = X[i, -1] - X[i, -2]  # 后向差分
    return derivative_X

# 读取训练集和测试集
train_path = "D:/A/CSU/数据集/CGL/CGL_nir_cal.xlsx"
test_path = "D:/A/CSU/数据集/CGL/CGL_nir_test.xlsx"

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train.iloc[:, :117].values
y_train = df_train.iloc[:, 120].values.ravel()
X_test = df_test.iloc[:, :117].values
y_test = df_test.iloc[:, 120].values.ravel()

# 预处理
X_train_sg = spectral_first_order_derivative(X_train)
X_test_sg = spectral_first_order_derivative(X_test)


# 特征选择
estimator = RandomForestRegressor(n_estimators=10, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=90, step=1)
X_train_rfe = rfe.fit_transform(X_train_sg, y_train)
X_test_rfe = rfe.transform(X_test_sg)

# TabPFN建模与预测
tabpfn_regressor = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42)
tabpfn_regressor.fit(X_train_rfe, y_train)
y_train_pred_tabpfn = tabpfn_regressor.predict(X_train_rfe)
y_test_pred_tabpfn = tabpfn_regressor.predict(X_test_rfe)

# PLSR建模与预测
param_grid = {'n_components': [5, 10, 15, 20]}
pls = PLSRegression()
grid = GridSearchCV(pls, param_grid, cv=3, scoring='neg_mean_squared_error')
grid.fit(X_train_rfe, y_train)
best_pls = grid.best_estimator_
y_train_pred_plsr = best_pls.predict(X_train_rfe).ravel()
y_test_pred_plsr = best_pls.predict(X_test_rfe).ravel()

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
    print(f"\n--- {name} 性能指标 ---")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"SEP: {sep:.4f}")

# 输出训练集和测试集性能
evaluate_performance(y_train, y_train_pred_tabpfn, "TabPFN-训练集")
evaluate_performance(y_test, y_test_pred_tabpfn, "TabPFN-测试集")
evaluate_performance(y_train, y_train_pred_plsr, "PLSR-训练集")
evaluate_performance(y_test, y_test_pred_plsr, "PLSR-测试集")

import matplotlib.pyplot as plt

def plot_error_distribution(y_true, y_pred, title):
    errors = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('pred errors')
    plt.ylabel('Samples')
    plt.title(title)
    plt.grid(True)
    plt.show()

# 可视化TabPFN和PLSR在测试集上的误差分布
plot_error_distribution(y_test, y_test_pred_tabpfn, "TabPFN test set error distribution")
plot_error_distribution(y_test, y_test_pred_plsr, "PLSR test set error distribution")