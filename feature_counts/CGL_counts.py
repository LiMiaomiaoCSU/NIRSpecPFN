import numpy as np
import pandas as pd
import time
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# experiment time
start_time = time.time()
experiment_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 读取训练集和测试集
train_path = "D:/A/CSU/数据集/CGL/CGL_nir_cal.xlsx"
test_path = "D:/A/CSU/数据集/CGL/CGL_nir_test.xlsx"

df_train = pd.read_excel(train_path)
df_test = pd.read_excel(test_path)

X_train = df_train.iloc[:, :117].values
y_train = df_train.iloc[:, 117].values.ravel()
X_test = df_test.iloc[:, :117].values
y_test = df_test.iloc[:, 117].values.ravel()

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

# 1. 先计算全谱TabPFN的RMSEP
X_train_sg = spectral_first_order_derivative(X_train)
X_test_sg = spectral_first_order_derivative(X_test)



tabpfn_full = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
tabpfn_full.fit(X_train_sg, y_train)
y_test_pred_full = tabpfn_full.predict(X_test_sg).ravel()
rmsep_full = np.sqrt(mean_squared_error(y_test, y_test_pred_full))
print(f"\n全谱TabPFN 测试集RMSEP: {rmsep_full:.4f}")

# 2. 不同RFE波长数下TabPFN
feature_nums = [50, 60, 70, 80, 90, 100, 110]
irmsep_results = []

print("\n--- 不同RFE波长数下TabPFN的RMSEP和iRMSEP ---")
for n_features in feature_nums:
    print(f"\n>>> RFE波长数: {n_features}")

    # RFE特征选择
    if n_features < X_train.shape[1]:
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        X_train_rfe = rfe.fit_transform(X_train_sg, y_train)
        X_test_rfe = rfe.transform(X_test_sg)
    else:
        X_train_rfe = X_train_sg
        X_test_rfe = X_test_sg

    tabpfn_rfe = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
    tabpfn_rfe.fit(X_train_rfe, y_train)
    y_test_pred_rfe = tabpfn_rfe.predict(X_test_rfe).ravel()
    rmsep_rfe = np.sqrt(mean_squared_error(y_test, y_test_pred_rfe))
    irmsep = calculate_irmsep(rmsep_full, rmsep_rfe)
    irmsep_results.append((n_features, rmsep_full, rmsep_rfe, irmsep))
    print(f"全谱RMSEP: {rmsep_full:.4f}  特征选择RMSEP: {rmsep_rfe:.4f}  iRMSEP: {irmsep:.2f}%")

# 保存结果
irmsep_df = pd.DataFrame(irmsep_results, columns=['RFE波长数', '全谱RMSEP', '特征选择RMSEP', 'iRMSEP(%)'])
result_file = "TabPFN_RFE_iRMSEP_results.xlsx"
irmsep_df.to_excel(result_file, index=False)
print(f"\n结果已保存到: {result_file}")