import numpy as np
import pandas as pd
import time
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from tabpfn import TabPFNRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing.process import derivative
from preprocessing.feature import rfe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# experiment time
start_time = time.time()
experiment_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load the data
data_path = "D:/A/CSU/数据集/wheat/Test_ManufacturerB.xlsx" #B instrument
df = pd.read_excel(data_path)
spectra = df.iloc[:, 2:1063].values  
y = df.iloc[:, 1].values.ravel()

# Evaluation metric
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
    
    print(f"\n--- {dataset_name} metrics ---")
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

print("\n--- K-fold cross ---")

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(spectra), 1):
    print(f"\n=== {fold} fold ===")
    
    X_train, X_test = spectra[train_idx], spectra[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    # Dataprocessing
    preprocess_method = 'Derivative'
    X_train_de = derivative(X_train)
    X_test_de = derivative(X_test)

    # Feature Selection
    feature_selection_method = 'RFE'
    X_train_rfe, X_test_rfe = rfe(X_train_de, y_train, X_test_de, n_features_to_select=600) #n_features_to_select can be adjusted

    # TabPFN-default
    tabpfn_regressor = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
    tabpfn_regressor.fit(X_train_rfe, y_train)
    y_train_pred_tabpfn = tabpfn_regressor.predict(X_train_rfe)
    y_test_pred_tabpfn = tabpfn_regressor.predict(X_test_rfe)
    tabpfn_train_results.append(evaluate_performance(y_train, y_train_pred_tabpfn, f"TabPFN-{fold}fold train"))
    tabpfn_test_results.append(evaluate_performance(y_test, y_test_pred_tabpfn, f"TabPFN-第{fold}fold test"))

    # PLSR-tuning
    param_grid = {'n_components': [5, 10, 15, 20]}  
    pls = PLSRegression()
    grid = GridSearchCV(pls, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train_rfe, y_train)
    best_pls = grid.best_estimator_
    print(f"PLSR best params: {grid.best_params_['n_components']}")
    y_train_pred_plsr = best_pls.predict(X_train_rfe).ravel()
    y_test_pred_plsr = best_pls.predict(X_test_rfe).ravel()
    plsr_train_results.append(evaluate_performance(y_train, y_train_pred_plsr, f"PLSR-{fold}fold train"))
    plsr_test_results.append(evaluate_performance(y_test, y_test_pred_plsr, f"PLSR-{fold}fold test"))
    
print("\n=== TabPFN 5 fold average metric ===")
print(pd.DataFrame(tabpfn_test_results).mean())
print("\n=== PLSR 5 fold average metric  ===")
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
    ['experiment time', experiment_start_time],
    ['elapsed_time(s)', f"{elapsed_time:.2f}"],
    ['device', device],
    ['data_path', data_path],
    ['samples', len(spectra)],
    ['spectra shape', spectra.shape[1]],
    ['y range', f"{y.min():.4f} - {y.max():.4f}"],
    ['y.mean', f"{y.mean():.4f}"],
    ['y.std', f"{y.std():.4f}"],
    ['preprocess_method', preprocess_method],
    ['feature_selection_method', feature_selection_method],
    ['', ''],
    ['model', 'dataset', 'R2-average±std', 'MAE-average±std', 'MSE-average±std', 'RMSE-average±std', 'SEP-average±std']
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
    df['set'] = ['train']*len(df) if 'train' in results[0].get('dataset_name', '') else ['test']*len(df)
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
    info_df.to_excel(writer, sheet_name='experiment information and average metric', header=False, index=False)
    tabpfn_detail.to_excel(writer, sheet_name='TabPFN', index=False)
    plsr_detail.to_excel(writer, sheet_name='PLSR', index=False)

print(f"\nresult_path: {result_file}") 
