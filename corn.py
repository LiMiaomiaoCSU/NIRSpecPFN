import numpy as np
import pandas as pd
import time
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# experiment time
start_time = time.time()
experiment_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load the data
data_path = 'D:/A/CSU/数据集/corn_xlsl/m5_corn.xlsx'

df = pd.read_excel(data_path)
spectra = df.iloc[:, :700].values  
y = df.iloc[:, 700].values.ravel()


def spectral_first_order_derivative(X):
        """光谱一阶微分处理"""
        derivative_X = np.zeros_like(X)
        for i in range(X.shape[0]):
        # 对每个样本进行一阶微分处理
        # 使用中心差分法，两端点使用前向和后向差分
            derivative_X[i, 0] = X[i, 1] - X[i, 0]  # 前向差分
            for j in range(1, X.shape[1] - 1):
                derivative_X[i, j] = (X[i, j + 1] - X[i, j - 1]) / 2  # 中心差分
            derivative_X[i, -1] = X[i, -1] - X[i, -2]  # 后向差分
        return derivative_X

def calculate_sep(y_true, y_pred):
    n = len(y_true)
    if n <= 1:
        return 0.0
    sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))
    return sep

def evaluate_performance(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    sep = calculate_sep(y_true, y_pred)
    
    print(f"\n--- {dataset_name} 性能指标 ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Standard Error of Prediction (SEP): {sep:.4f}")
    
    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse, 'sep': sep}

def calculate_average_metrics(results_list):
    avg_metrics = {}
    for metric in ['r2', 'mae', 'mse', 'rmse', 'sep']:
        values = [result[metric] for result in results_list]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    return avg_metrics

tabpfn_train_results, tabpfn_test_results = [], []
plsr_train_results, plsr_test_results = [], []

print("\n--- 开始5折交叉验证 ---")

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(spectra), 1):
    print(f"\n=== 第 {fold} 折 ===")
    
    X_train, X_test = spectra[train_idx], spectra[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # Dataprocessing
    preprocess_method = 'Derivative'
    X_train_sg = spectral_first_order_derivative(X_train)
    X_test_sg = spectral_first_order_derivative(X_test)

    # Feature Selection
    feature_selection_method = 'RFE'
    estimator = RandomForestRegressor(n_estimators=10, random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=150, step=1)
    X_train_rfe = rfe.fit_transform(X_train_sg, y_train)
    X_test_rfe = rfe.transform(X_test_sg)

    # TabPFN-default
    tabpfn_regressor = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
    tabpfn_regressor.fit(X_train_rfe, y_train)
    y_train_pred_tabpfn = tabpfn_regressor.predict(X_train_rfe)
    y_test_pred_tabpfn = tabpfn_regressor.predict(X_test_rfe)
    tabpfn_train_results.append(evaluate_performance(y_train, y_train_pred_tabpfn, f"TabPFN-第{fold}折训练集"))
    tabpfn_test_results.append(evaluate_performance(y_test, y_test_pred_tabpfn, f"TabPFN-第{fold}折测试集"))

    # PLSR-tuning
    param_grid = {'n_components': [5, 10, 15, 20]}  
    pls = PLSRegression()
    grid = GridSearchCV(pls, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train_rfe, y_train)
    best_pls = grid.best_estimator_
    print(f"PLSR最佳主成分数: {grid.best_params_['n_components']}")
    y_train_pred_plsr = best_pls.predict(X_train_rfe).ravel()
    y_test_pred_plsr = best_pls.predict(X_test_rfe).ravel()
    plsr_train_results.append(evaluate_performance(y_train, y_train_pred_plsr, f"PLSR-第{fold}折训练集"))
    plsr_test_results.append(evaluate_performance(y_test, y_test_pred_plsr, f"PLSR-第{fold}折测试集"))
    
print("\n=== TabPFN模型5折平均性能 ===")
print(pd.DataFrame(tabpfn_test_results).mean())
print("\n=== PLSR模型5折平均性能 ===")
print(pd.DataFrame(plsr_test_results).mean())
    
end_time = time.time()
elapsed_time = end_time - start_time

# Calculate average metrics
tabpfn_train_avg = calculate_average_metrics(tabpfn_train_results)
tabpfn_test_avg = calculate_average_metrics(tabpfn_test_results)
plsr_train_avg = calculate_average_metrics(plsr_train_results)
plsr_test_avg = calculate_average_metrics(plsr_test_results)

# Save results
info_rows = [
    ['实验时间', experiment_start_time],
    ['所需时间（秒）', f"{elapsed_time:.2f}"],
    ['使用设备', device],
    ['数据集路径', data_path],
    ['总样本数', len(spectra)],
    ['原始特征维度', spectra.shape[1]],
    ['标签范围', f"{y.min():.4f} - {y.max():.4f}"],
    ['标签均值', f"{y.mean():.4f}"],
    ['标签标准差', f"{y.std():.4f}"],
    ['预处理方法', preprocess_method],
    ['特征选择方法', feature_selection_method],
    ['', ''],
    ['模型', '数据集', 'R2均值±std', 'MAE均值±std', 'MSE均值±std', 'RMSE均值±std', 'SEP均值±std']
]

def avg_row(avg, model, dataset):
    return [
        model, dataset,
        f"{avg['r2']['mean']:.4f}±{avg['r2']['std']:.4f}",
        f"{avg['mae']['mean']:.4f}±{avg['mae']['std']:.4f}",
        f"{avg['mse']['mean']:.4f}±{avg['mse']['std']:.4f}",
        f"{avg['rmse']['mean']:.4f}±{avg['rmse']['std']:.4f}",
        f"{avg['sep']['mean']:.4f}±{avg['sep']['std']:.4f}"
    ]

info_rows.append(avg_row(tabpfn_train_avg, 'TabPFN', 'Train'))
info_rows.append(avg_row(tabpfn_test_avg, 'TabPFN', 'Test'))
info_rows.append(avg_row(plsr_train_avg, 'PLSR', 'Train'))
info_rows.append(avg_row(plsr_test_avg, 'PLSR', 'Test'))

info_df = pd.DataFrame(info_rows)

def detail_df(results, model_name):
    df = pd.DataFrame(results)
    df['fold'] = np.arange(1, len(df)+1)
    df['set'] = ['train']*len(df) if '训练集' in results[0].get('dataset_name', '') else ['test']*len(df)
    if 'dataset_name' in results[0]:
        df['dataset_name'] = [r['dataset_name'] for r in results]
    return df

tabpfn_detail = pd.concat([
    pd.DataFrame([{**r, 'fold': i+1, 'set': 'train'} for i, r in enumerate(tabpfn_train_results)]),
    pd.DataFrame([{**r, 'fold': i+1, 'set': 'test'} for i, r in enumerate(tabpfn_test_results)])
], ignore_index=True)

plsr_detail = pd.concat([
    pd.DataFrame([{**r, 'fold': i+1, 'set': 'train'} for i, r in enumerate(plsr_train_results)]),
    pd.DataFrame([{**r, 'fold': i+1, 'set': 'test'} for i, r in enumerate(plsr_test_results)])
], ignore_index=True)



data_file_base = os.path.basename(data_path)         # data_file
data_file_tag = data_file_base.split('_')[0]         # data_file_tag
y_col_num = [i for i, col in enumerate(df.columns) if np.array_equal(df[col].values.ravel(), y)][0]
y_col_index = df.columns[y_col_num]    # column index of label                    
# Namespace for result file
result_file = f"{data_file_tag}_y{y_col_index}_{preprocess_method}_{feature_selection_method}_results.xlsx"

with pd.ExcelWriter(result_file) as writer:
    info_df.to_excel(writer, sheet_name='实验信息与平均指标', header=False, index=False)
    tabpfn_detail.to_excel(writer, sheet_name='TabPFN', index=False)
    plsr_detail.to_excel(writer, sheet_name='PLSR', index=False)

print(f"\n结果已保存到: {result_file}") 