import numpy as np
import pandas as pd
import time
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# experiment time
start_time = time.time()
experiment_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load the data
data_path = "D:/A/CSU/数据集/wheat/Test_ManufacturerB.xlsx"
df = pd.read_excel(data_path)
spectra = df.iloc[:, 2:1063].values  
y = df.iloc[:, 1].values.ravel()

def spectral_first_order_derivative(X):
    """光谱一阶微分处理"""
    derivative_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        derivative_X[i, 0] = X[i, 1] - X[i, 0]  # 前向差分
        for j in range(1, X.shape[1] - 1):
            derivative_X[i, j] = (X[i, j + 1] - X[i, j - 1]) / 2  # 中心差分
        derivative_X[i, -1] = X[i, -1] - X[i, -2]  # 后向差分
    return derivative_X

def calculate_irmsep(full_rmsep, selected_rmsep):
    return ((full_rmsep - selected_rmsep) / full_rmsep) * 100

# 1. 先计算全谱TabPFN的8折平均RMSEP
kf = KFold(n_splits=8, shuffle=True, random_state=42)
rmsep_full_list = []
for fold, (train_idx, test_idx) in enumerate(kf.split(spectra), 1):
    X_train, X_test = spectra[train_idx], spectra[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 一阶微分
    X_train_sg = spectral_first_order_derivative(X_train)
    X_test_sg = spectral_first_order_derivative(X_test)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sg)
    X_test_scaled = scaler.transform(X_test_sg)

    # 全谱TabPFN
    tabpfn_full = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
    tabpfn_full.fit(X_train_scaled, y_train)
    y_test_pred_full = tabpfn_full.predict(X_test_scaled).ravel()
    rmse_full = np.sqrt(mean_squared_error(y_test, y_test_pred_full))
    rmsep_full_list.append(rmse_full)

avg_full_rmsep = np.mean(rmsep_full_list)
print(f"\n全谱TabPFN 5折平均RMSEP: {avg_full_rmsep:.4f}")

# 2. 不同RFE波长数下TabPFN
feature_nums = [50, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600, 650, 700]
irmsep_results = []

print("\n--- 不同RFE波长数下TabPFN的RMSEP和iRMSEP ---")
for n_features in feature_nums:
    print(f"\n>>> RFE波长数: {n_features}")
    rmsep_rfe_list = []
    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(spectra), 1):
        X_train, X_test = spectra[train_idx], spectra[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 一阶微分
        X_train_sg = spectral_first_order_derivative(X_train)
        X_test_sg = spectral_first_order_derivative(X_test)

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sg)
        X_test_scaled = scaler.transform(X_test_sg)

        # RFE特征选择
        if n_features < spectra.shape[1]:
            estimator = RandomForestRegressor(n_estimators=10, random_state=42)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
            X_test_rfe = rfe.transform(X_test_scaled)
        else:
            X_train_rfe = X_train_scaled
            X_test_rfe = X_test_scaled

        tabpfn_rfe = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
        tabpfn_rfe.fit(X_train_rfe, y_train)
        y_test_pred_rfe = tabpfn_rfe.predict(X_test_rfe).ravel()
        rmse_rfe = np.sqrt(mean_squared_error(y_test, y_test_pred_rfe))
        rmsep_rfe_list.append(rmse_rfe)

    avg_rfe_rmsep = np.mean(rmsep_rfe_list)
    irmsep = calculate_irmsep(avg_full_rmsep, avg_rfe_rmsep)
    irmsep_results.append((n_features, avg_full_rmsep, avg_rfe_rmsep, irmsep))
    print(f"全谱RMSEP: {avg_full_rmsep:.4f}  特征选择RMSEP: {avg_rfe_rmsep:.4f}  iRMSEP: {irmsep:.2f}%")

# 保存结果
irmsep_df = pd.DataFrame(irmsep_results, columns=['RFE波长数', '全谱RMSEP', '特征选择RMSEP', 'iRMSEP(%)'])
result_file = "TabPFN_RFE_iRMSEP_results.xlsx"
irmsep_df.to_excel(result_file, index=False)
print(f"\n结果已保存到: {result_file}")