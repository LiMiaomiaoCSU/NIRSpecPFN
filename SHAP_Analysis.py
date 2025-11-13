import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNRegressor
from preprocessing.process import derivative
from preprocessing.feature import rfe
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.colors as mcolors


device = 'cpu'
# Load the data
data_path = 'D:/A/CSU/corn_xlsl/m5_corn.xlsx'

df = pd.read_excel(data_path)
spectra = df.iloc[:, :700].values  
y = df.iloc[:, 701].values.ravel() #Protein Content

X_tr, X_te, y_tr, y_te = train_test_split(spectra, y, test_size=0.2, random_state=42, shuffle=True)

X_tr_de = derivative(X_tr)
X_te_de = derivative(X_te)

X_tr_rfe, X_te_rfe = rfe(X_tr_de, y_tr, X_te_de)

model = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True)
model.fit(X_tr_rfe, y_tr)

def map_selected_indices(X_orig, X_rfe):
    n_sel = X_rfe.shape[1]
    sel = []
    used = set()
    for i in range(n_sel):
        col = X_rfe[:, i]
        errs = np.mean((X_orig - col[:, None])**2, axis=0)
        idx = int(np.argmin(errs))
        if idx in used:
            for j in np.argsort(errs):
                if int(j) not in used:
                    idx = int(j)
                    break
        sel.append(idx)
        used.add(idx)
    return np.array(sel, dtype=int)

sel_idx = map_selected_indices(X_tr_de, X_tr_rfe)

wave = np.arange(1100, 2498 + 2, 2)  # 700 channels
wave_sel = wave[sel_idx]
feat_names = [f"{int(w)}nm" for w in wave_sel]

if "__file__" in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    base_dir = os.getcwd()          
out_dir = os.path.join(base_dir, "701_shap_results")
os.makedirs(out_dir, exist_ok=True)
print("SHAP path：", out_dir)

bg_n = min(30, X_tr_rfe.shape[0])
nt = min(50, X_te_rfe.shape[0])

background = X_tr_rfe[:bg_n]

def _predict(X):
    return np.ravel(model.predict(np.asarray(X)))

explainer = shap.Explainer(_predict, background)
X_sub = X_te_rfe[:nt, :]
expl = explainer(X_sub)

top_features = min(10, X_tr_rfe.shape[1])
shap_values = expl.values if hasattr(expl, 'values') else np.array(expl)
feature_importance = np.mean(np.abs(shap_values), axis=0)
top_indices = np.argsort(feature_importance)[-top_features:][::-1]
    
print("Top features for dependency analysis:")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {feat_names[idx]} (importance: {feature_importance[idx]:.4f})")
    
# summary bar
plt.figure(figsize=(8,4))
try:
    shap.summary_plot(expl, plot_type="bar", feature_names=feat_names, show=False)
except Exception:
    vals = expl.values if hasattr(expl, "values") else np.array(expl)
    shap.summary_plot(vals, X_sub, plot_type="bar", feature_names=feat_names, show=False)
plt.title("SHAP global importance")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=300)
plt.close()

# summary dot
plt.figure(figsize=(8,6))
try:
    shap.summary_plot(expl, feature_names=feat_names, show=False)
except Exception:
    vals = expl.values if hasattr(expl, "values") else np.array(expl)
    shap.summary_plot(vals, X_sub, feature_names=feat_names, show=False)
plt.title("SHAP summary (dot)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "shap_dot.png"), dpi=300)
plt.close()


print("all files path：", out_dir)
