import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning, module="tabpfn.*")

# ==============================================================
# 1. Configuration
# ==============================================================
file_path = r"D:\A\CSU\NIRdatasets\tecator\preprocessed_tecator.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

target_property = "fat"        # "moisture", "fat", "protein"
selection_mode = "per_model"      # "per_model" or "global"
all_preprocessing_sheets = [
    "Raw", "MSC", "SNV",
    "First Derivative", "SG-2D", "airPLS"
]

BASELINE_MODEL = "NIRSpecPFN"
model_names = ["NIRSpecPFN", "PLSR", "SVR", "RF"]

output_dir = os.path.join(
    os.path.dirname(__file__),
    f"tecator_{target_property}_sep_sig"
)
os.makedirs(output_dir, exist_ok=True)

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", tabpfn_cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(tabpfn_cache_dir, "tabpfn-v2.5-regressor-v2.5_real.ckpt")


# ==============================================================
# 2. Evaluation helpers
# ==============================================================
def calculate_sep(y_true, y_pred):
    """Standard Error of Prediction (SEP), denominator n-1."""
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 1))


def evaluate_cv_rmsecv(model_name, X_tr, y_tr, seed):
    """Return RMSECV and best hyperparameters on training set."""
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)

    if model_name == "NIRSpecPFN":
        fold_rmses = []
        for tr_idx, va_idx in cv_inner.split(X_tr):
            model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
            setattr(model, "ignore_pretraining_limits", True)
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            preds = model.predict(X_tr[va_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(y_tr[va_idx], preds)))
        return {
            "RMSECV": float(np.mean(fold_rmses)),
            "Best_n": np.nan, "Best_C": np.nan, "Best_epsilon": np.nan,
        }

    if model_name == "PLSR":
        grid = GridSearchCV(
            PLSRegression(), {"n_components": list(range(5, 16))},
            cv=cv_inner, scoring="neg_mean_squared_error"
        )
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": grid.best_params_["n_components"],
            "Best_C": np.nan, "Best_epsilon": np.nan,
        }

    if model_name == "SVR":
        grid = GridSearchCV(
            SVR(),
            {"C": [100, 200, 300, 400, 500],
             "epsilon": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
             "kernel": ["rbf"]},
            cv=cv_inner, scoring="neg_mean_squared_error"
        )
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": np.nan,
            "Best_C": grid.best_params_["C"],
            "Best_epsilon": grid.best_params_["epsilon"],
        }

    if model_name == "RF":
        grid = GridSearchCV(
            RandomForestRegressor(random_state=seed),
            {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": [1, 2, 4],
            },
            cv=cv_inner, scoring="neg_mean_squared_error"
        )
        grid.fit(X_tr, y_tr)
        return {
            "RMSECV": np.sqrt(-grid.best_score_),
            "Best_n": grid.best_params_["n_estimators"],
            "Best_C": grid.best_params_["max_depth"],
            "Best_epsilon": grid.best_params_["min_samples_leaf"],
        }

    raise ValueError(f"Unsupported model: {model_name}")

def evaluate_test(model_name, X_tr, y_tr, X_te, y_te, seed,
                  best_n=np.nan, best_c=np.nan, best_epsilon=np.nan):
    """
    Fit on full training set and evaluate on fixed test set.
    Returns RMSE, SEP, R², MAE, RPD, and sample-level SE.
    """
    if model_name == "NIRSpecPFN":
        model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
        setattr(model, "ignore_pretraining_limits", True)
    elif model_name == "PLSR":
        model = PLSRegression(n_components=int(best_n))
    elif model_name == "SVR":
        model = SVR(C=float(best_c), epsilon=float(best_epsilon), kernel="rbf")
    elif model_name == "RF":
        model = RandomForestRegressor(n_estimators=int(best_n), max_depth=best_c, min_samples_leaf=best_epsilon, random_state=seed)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    if hasattr(preds, "ravel"):
        preds = preds.ravel()

    residuals = y_te - preds
    n = len(y_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    sep = float(np.sqrt(np.sum(residuals ** 2) / (n - 1))) if n > 1 else rmse
    r2 = float(r2_score(y_te, preds))
    mae = float(mean_absolute_error(y_te, preds))
    rpd = float(np.std(y_te, ddof=1) / rmse) if rmse != 0 else np.nan

    
    se = residuals ** 2

    return {
        "RMSE": rmse, "SEP": sep, "R2": r2, "MAE": mae, "RPD": rpd,
        "SE": se, "residuals": residuals, "preds": np.asarray(preds, dtype=float),
    }


# ==============================================================
# 3. FIXED predefined split (C+M = train, T = test)
# ==============================================================
print(f"\n>>> FIXED predefined split for tecator ({target_property})")

dataset_cache = {}
for sheet_name in all_preprocessing_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    subset = df.iloc[:, 1]
    train_mask = subset.isin(["C", "M"])
    test_mask = subset == "T"

    X = df.iloc[:, 2:102].to_numpy()
    y = df[target_property].to_numpy()

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  Warning: sheet '{sheet_name}' is empty, skipping.")
        continue

    dataset_cache[sheet_name] = (X_train, X_test, y_train, y_test)
    print(f"  {sheet_name}: train={len(X_train)}, test={len(X_test)}")


# ==============================================================
# 4. Preprocessing selection (RMSECV on full training set)
# ==============================================================
print("\n>>> Preprocessing selection (RMSECV on full training set)")
best_per_model = {}

for model_name in model_names:
    best_rmsecv = np.inf
    best_cfg = None
    for sheet_name, (X_tr, _, y_tr, _) in dataset_cache.items():
        cv_res = evaluate_cv_rmsecv(model_name, X_tr, y_tr, seed=42)
        if cv_res["RMSECV"] < best_rmsecv:
            best_rmsecv = cv_res["RMSECV"]
            best_cfg = {
                "Preprocessing": sheet_name,
                "RMSECV": cv_res["RMSECV"],
                "Best_n": cv_res["Best_n"],
                "Best_C": cv_res["Best_C"],
                "Best_epsilon": cv_res["Best_epsilon"],
            }
    best_per_model[model_name] = best_cfg
    print(f"{model_name}: {best_cfg['Preprocessing']}, RMSECV={best_cfg['RMSECV']:.4f}")


# ==============================================================
# 5. Final test evaluation (single pass, fixed test set)
# ==============================================================
print("\n>>> Final test evaluation on fixed test set")
final_records = []
se_by_model = {}          # sample-level squared errors
preds_by_model = {}       # predictions

for model_name in model_names:
    cfg = best_per_model[model_name]
    X_tr, X_te, y_tr, y_te = dataset_cache[cfg["Preprocessing"]]
    res = evaluate_test(
        model_name, X_tr, y_tr, X_te, y_te, seed=42,
        best_n=cfg["Best_n"], best_c=cfg["Best_C"], best_epsilon=cfg["Best_epsilon"]
    )
    se_by_model[model_name] = res["SE"]
    preds_by_model[model_name] = res["preds"]

    final_records.append({
        "Model": model_name,
        "Preprocessing": cfg["Preprocessing"],
        "Train_RMSECV": cfg["RMSECV"],
        "Test_R2": res["R2"],
        "Test_RMSE": res["RMSE"],
        "Test_MAE": res["MAE"],
        "Test_SEP": res["SEP"],
        "Test_RPD": res["RPD"],
    })
    print(f"{model_name}: R²={res['R2']:.2f}, RMSE={res['RMSE']:.2f}, "
          f"MAE={res['MAE']:.2f}, SEP={res['SEP']:.2f}, RPD={res['RPD']:.2f}")

final_df = pd.DataFrame(final_records)


# ==============================================================
# 6. Paired significance test on sample-level SE (proxy for SEP)
# ==============================================================
print(f"\n>>> Paired significance test on sample-level SE (proxy for SEP)")

se_tab = se_by_model["NIRSpecPFN"]
se_plsr = se_by_model["PLSR"]
se_svr = se_by_model["SVR"]
se_rf = se_by_model["RF"]



diff_tab_vs_plsr = se_tab - se_plsr
diff_tab_vs_svr = se_tab - se_svr
diff_tab_vs_rf = se_tab - se_rf

def paired_test(diff_seq):
    """Shapiro-Wilk + paired t-test or Wilcoxon signed-rank test."""
    diff_seq = np.asarray(diff_seq, dtype=float)
    shapiro_stat, shapiro_p = shapiro(diff_seq)
    if shapiro_p > 0.05:
        test_name = "paired_t_test"
        test_stat, p_value = ttest_rel(diff_seq, np.zeros_like(diff_seq))
    else:
        test_name = "wilcoxon"
        test_stat, p_value = wilcoxon(diff_seq)
    return {
        "Shapiro_W": float(shapiro_stat), "Shapiro_p": float(shapiro_p),
        "Test_Name": test_name, "Test_Stat": float(test_stat), "P_Value": float(p_value),
    }


def sig_label(p):
    """ᵃ(p≤0.005) ᵇ(p≤0.01) ᶜ(p≤0.05)"""
    if p <= 0.005: return "ᵃ"
    if p <= 0.01:  return "ᵇ"
    if p <= 0.05:  return "ᶜ"
    return ""


sig_tab_vs_plsr = paired_test(diff_tab_vs_plsr)
sig_tab_vs_svr = paired_test(diff_tab_vs_svr)
sig_tab_vs_rf = paired_test(diff_tab_vs_rf)
print(f"NIRSpecPFN vs PLSR: p={sig_tab_vs_plsr['P_Value']:.6f} ({sig_tab_vs_plsr['Test_Name']})")
print(f"NIRSpecPFN vs SVR : p={sig_tab_vs_svr['P_Value']:.6f} ({sig_tab_vs_svr['Test_Name']})")
print(f"NIRSpecPFN vs RF  : p={sig_tab_vs_rf['P_Value']:.6f} ({sig_tab_vs_rf['Test_Name']})")



# Build significance table (baseline-centric, only mark when model is worse)
sig_results = []
for model, diff, sig_res in [("PLSR", diff_tab_vs_plsr, sig_tab_vs_plsr),
                               ("SVR", diff_tab_vs_svr, sig_tab_vs_svr),
                               ("RF", diff_tab_vs_rf, sig_tab_vs_rf)]:
    mean_diff = np.mean(diff)
    p = sig_res["P_Value"]
    label = ""
    if mean_diff > 0 and p <= 0.05:   # model is significantly worse than baseline
        label = sig_label(p)
    sig_results.append({
        "Model": model,
        "Test_R2": final_df[final_df["Model"] == model]["Test_R2"].values[0],
        "Test_RMSE": final_df[final_df["Model"] == model]["Test_RMSE"].values[0],
        "Test_SEP": final_df[final_df["Model"] == model]["Test_SEP"].values[0],
        "Mean_SE_Diff": float(mean_diff),
        "Shapiro_p": sig_res["Shapiro_p"],
        "Test": sig_res["Test_Name"],
        "P_Value": p,
        "Sig": label,
    })

# Baseline row
baseline_r2 = final_df[final_df["Model"] == BASELINE_MODEL]["Test_R2"].values[0]
baseline_rmse = final_df[final_df["Model"] == BASELINE_MODEL]["Test_RMSE"].values[0]
baseline_sep = final_df[final_df["Model"] == BASELINE_MODEL]["Test_SEP"].values[0]
sig_results.insert(0, {
    "Model": BASELINE_MODEL,
    "Test_R2": baseline_r2,
    "Test_RMSE": baseline_rmse,
    "Test_SEP": baseline_sep,
    "Mean_SE_Diff": 0.0,
    "Shapiro_p": np.nan,
    "Test": "—",
    "P_Value": np.nan,
    "Sig": "",
})

sig_df = pd.DataFrame(sig_results)
print("\n>>> Significance summary")
print(sig_df.to_string(index=False))


# ==============================================================
# 7. Save results
# ==============================================================
final_csv = os.path.join(output_dir, f"{target_property}_final_metrics.csv")
final_df.to_csv(final_csv, index=False)

sig_csv = os.path.join(output_dir, f"{target_property}_significance.csv")
sig_df.to_csv(sig_csv, index=False)

se_df = pd.DataFrame({
    "Test_Index": np.arange(len(se_tab)),
    "y_true": y_te,
    "NIRSpecPFN_SE": se_by_model["NIRSpecPFN"],
    "PLSR_SE": se_by_model["PLSR"],
    "SVR_SE": se_by_model["SVR"],
    "RF_SE": se_by_model["RF"],
    "Diff_Tab_vs_PLSR": diff_tab_vs_plsr,
    "Diff_Tab_vs_SVR": diff_tab_vs_svr,
    "Diff_Tab_vs_RF": diff_tab_vs_rf,
})
se_csv = os.path.join(output_dir, f"{target_property}_sample_se.csv")
se_df.to_csv(se_csv, index=False)

print(f"\nResults saved to: {output_dir}")


# ==============================================================
# 8. Plot: sample-level SE distribution + scatter plots
# ==============================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# 8a. SE boxplot
fig, ax = plt.subplots(figsize=(6, 5))
data_to_plot = [se_plsr, se_svr, se_tab, se_rf]
bp = ax.boxplot(data_to_plot, tick_labels=["PLSR", "SVR", "NIRSpecPFN", "RF"],
                whis=3.0, showfliers=False)
ax.set_ylabel("Sample-level Squared Error (SE)", fontsize=18)
ax.set_title(f"Tecator {target_property}: SE Distribution\n(SE ∝ SEP², fixed test set)", fontsize=15)
ax.tick_params(axis="both", which="major", labelsize=16)

y_max = max(d.max() for d in data_to_plot)
y_min = min(d.min() for d in data_to_plot)
yr = max(y_max - y_min, 1e-6)

for i, m in enumerate(["PLSR", "SVR", "NIRSpecPFN", "RF"], 1):
    rmse = final_df[final_df["Model"] == m]["Test_RMSE"].values[0]
    sep = final_df[final_df["Model"] == m]["Test_SEP"].values[0]
    sig = sig_df[sig_df["Model"] == m]["Sig"].values[0] if m in sig_df["Model"].values else ""
    ax.text(i, y_min - 0.03 * yr,
            f"RMSE={rmse:.3f}\nSEP={sep:.3f}{sig}",
            ha="center", va="top", fontsize=10, color="navy")

def add_sig_bar(ax, x1, x2, y, h, text, lw=1.2):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], c="black", lw=lw)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=11)

h = 0.03 * yr
y_pos = y_max + 0.05 * yr
for _, row in sig_df.iterrows():
    if row["Model"] == BASELINE_MODEL or not row["Sig"]:
        continue
    x_base = 3  # NIRSpecPFN
    x_comp = 1 if row["Model"] == "PLSR" else 2
    add_sig_bar(ax, x_base, x_comp, y_pos, h, row["Sig"])
    y_pos += 0.12 * yr

ax.set_ylim(y_min - 0.15 * yr, y_pos + 0.05 * yr)
ax.text(0.02, 0.98, "ᵃ p ≤ 0.005\nᵇ p ≤ 0.01\nᶜ p ≤ 0.05",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="0.8"))
plt.tight_layout()
plot_path = os.path.join(output_dir, f"se_distribution_{target_property}.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"SE plot saved to: {plot_path}")

# 8b. True vs Predicted scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
y_min = min(np.min(y_te), *(np.min(preds_by_model[m]) for m in model_names))
y_max = max(np.max(y_te), *(np.max(preds_by_model[m]) for m in model_names))

for ax, model_name in zip(axes, model_names):
    y_pred = preds_by_model[model_name]
    ax.scatter(y_te, y_pred, alpha=0.75)
    ax.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=1.2)
    ax.set_title(f"{model_name} | {best_per_model[model_name]['Preprocessing']}", fontsize=20)
    ax.set_xlabel("True", fontsize=20)
    ax.set_ylabel("Predicted", fontsize=20)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=16)
    r2 = final_df[final_df["Model"] == model_name]["Test_R2"].values[0]
    rmse = final_df[final_df["Model"] == model_name]["Test_RMSE"].values[0]
    ax.text(0.05, 0.95, f"R²: {r2:.4f}\nRMSEP: {rmse:.4f}",
            transform=ax.transAxes, fontsize=18, verticalalignment="top")

scatter_path = os.path.join(output_dir, f"scatter_{target_property}.png")
fig.savefig(scatter_path, dpi=300)
plt.show()
print(f"Scatter plot saved to: {scatter_path}")
print("\n>>> Done.")