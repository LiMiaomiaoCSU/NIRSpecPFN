import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kennard_stone import train_test_split as ks_split
from sklearn.model_selection import train_test_split


from scipy.stats import shapiro, ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning, module="tabpfn.*")

# ==============================================================
# 1. Configuration
# ==============================================================
file_path = r"D:\A\CSU\NIRdatasets\soil\preprocessed_soil.xlsx"
tabpfn_cache_dir = r"D:\workspace\TabPFN\tabpfn"

target_property = "TC (%)"          # "OM (%)" or "TC (%)"
all_preprocessing_sheets = [
    "Raw spectral", "Smoothing", "MSC", "SNV",
    "1-derivative", "2-derivative", "Mean center", "S-G+SNV",
]

BASELINE_MODEL = "NIRSpecPFN"       
model_names = ["NIRSpecPFN", "PLSR", "SVR", "RF"]

output_dir = os.path.join(
    os.path.dirname(__file__),
    f"soil_{target_property}_sep_sig"
)
os.makedirs(output_dir, exist_ok=True)

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", tabpfn_cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(tabpfn_cache_dir, "tabpfn-v2.5-regressor-v2.5_real.ckpt")


# ==============================================================
# 2. Evaluation helpers
# ==============================================================
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
    Returns RMSE, SEP, R², and sample-level squared errors (SE).
    """
    if model_name == "NIRSpecPFN":
        model = TabPFNRegressor(model_path=model_path, device=device, random_state=seed)
        setattr(model, "ignore_pretraining_limits", True)
    elif model_name == "PLSR":
        model = PLSRegression(n_components=int(best_n))
    elif model_name == "SVR":
        model = SVR(C=float(best_c), epsilon=float(best_epsilon), kernel="rbf")
    elif model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=int(best_n),
            max_depth=None if pd.isna(best_c) else int(best_c),
            min_samples_leaf=int(best_epsilon),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    if hasattr(preds, "ravel"):
        preds = preds.ravel()

    residuals = y_te - preds
    n = len(y_te)
    mse = np.mean(residuals ** 2)
    rmse = float(np.sqrt(mse))
    # SEP: standard error of prediction 
    sep = float(np.sqrt(np.sum(residuals ** 2) / (n - 1))) if n > 1 else rmse

   
    se = residuals ** 2

    return {
        "RMSE": rmse,
        "SEP": sep,
        "R2": r2_score(y_te, preds),
        "SE": se,                # sample-level squared errors
        "residuals": residuals,
        "preds": np.asarray(preds, dtype=float),
    }


# ==============================================================
# 3. FIXED Kennard-Stone split (deterministic, never changes)
# ==============================================================
print("\n>>> FIXED K-S split (deterministic)")
df_ref = pd.read_excel(file_path, sheet_name=all_preprocessing_sheets[0])
X_ref = df_ref.iloc[:, 1:-2].to_numpy()
y_ref = df_ref[target_property].to_numpy()

indices = np.arange(len(X_ref)).reshape(-1, 1)
X_with_idx = np.hstack([X_ref, indices])
X_train_idx, X_test_idx, y_train, y_test = train_test_split(X_with_idx, y_ref, test_size=0.3, random_state=182)

train_idx = X_train_idx[:, -1].astype(int)
test_idx = X_test_idx[:, -1].astype(int)
print(f"Training: {len(train_idx)}, Test: {len(test_idx)}")

# Cache all preprocessing sheets with FIXED indices
dataset_cache = {}
for sheet in all_preprocessing_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)
    X_all = df.iloc[:, 1:-2].to_numpy()
    dataset_cache[sheet] = (X_all[train_idx], X_all[test_idx])

y_train_fixed = y_ref[train_idx]
y_test_fixed = y_ref[test_idx]


# ==============================================================
# 4. Preprocessing selection (on FULL training set, done once)
# ==============================================================
print("\n>>> Preprocessing selection (RMSECV on full training set)")
best_per_model = {}

for model_name in model_names:
    best_rmsecv = np.inf
    best_cfg = None
    for sheet in all_preprocessing_sheets:
        X_tr, _ = dataset_cache[sheet]
        cv_res = evaluate_cv_rmsecv(model_name, X_tr, y_train_fixed, seed=42)
        if cv_res["RMSECV"] < best_rmsecv:
            best_rmsecv = cv_res["RMSECV"]
            best_cfg = {
                "Preprocessing": sheet,
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
residuals_by_model = {}   # raw residuals

for model_name in model_names:
    cfg = best_per_model[model_name]
    X_tr, X_te = dataset_cache[cfg["Preprocessing"]]
    res = evaluate_test(
        model_name, X_tr, y_train_fixed, X_te, y_test_fixed, seed=42,
        best_n=cfg["Best_n"], best_c=cfg["Best_C"], best_epsilon=cfg["Best_epsilon"]
    )
    se_by_model[model_name] = res["SE"]
    residuals_by_model[model_name] = res["residuals"]

    final_records.append({
        "Model": model_name,
        "Preprocessing": cfg["Preprocessing"],
        "Train_RMSECV": cfg["RMSECV"],
        "Test_RMSE": res["RMSE"],
        "Test_SEP": res["SEP"],
    })
    print(f"{model_name}: RMSE={res['RMSE']:.4f}, SEP={res['SEP']:.4f}")

final_df = pd.DataFrame(final_records)


# ==============================================================
# 6. Paired significance test on sample-level squared errors (SE)
#    SE is the sample-level component of SEP²; testing SE differences
#    is mathematically equivalent to testing SEP differences.
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

print(f"NIRSpecPFN vs PLSR: p={sig_tab_vs_plsr['P_Value']:.2f} ({sig_tab_vs_plsr['Test_Name']})")
print(f"NIRSpecPFN vs SVR : p={sig_tab_vs_svr['P_Value']:.2f} ({sig_tab_vs_svr['Test_Name']})")
print(f"NIRSpecPFN vs RF  : p={sig_tab_vs_rf['P_Value']:.2f} ({sig_tab_vs_rf['Test_Name']})")

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
        "Test_RMSE": final_df[final_df["Model"] == model]["Test_RMSE"].values[0],
        "Test_SEP": final_df[final_df["Model"] == model]["Test_SEP"].values[0],
        "Mean_SE_Diff": float(mean_diff),
        "Shapiro_p": sig_res["Shapiro_p"],
        "Test": sig_res["Test_Name"],
        "P_Value": p,
        "Sig": label,
    })

# Baseline row
baseline_rmse = final_df[final_df["Model"] == BASELINE_MODEL]["Test_RMSE"].values[0]
baseline_sep = final_df[final_df["Model"] == BASELINE_MODEL]["Test_SEP"].values[0]
sig_results.insert(0, {
    "Model": BASELINE_MODEL,
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

# Save sample-level SE for inspection
se_df = pd.DataFrame({
    "Test_Index": np.arange(len(y_test_fixed)),
    "y_true": y_test_fixed,
    "NIRSpecPFN_SE": se_by_model["NIRSpecPFN"],
    "PLSR_SE": se_by_model["PLSR"],
    "SVR_SE": se_by_model["SVR"],
    "Diff_Tab_vs_PLSR": diff_tab_vs_plsr,
    "Diff_Tab_vs_SVR": diff_tab_vs_svr,
    "Diff_Tab_vs_RF": diff_tab_vs_rf,
})
se_csv = os.path.join(output_dir, f"{target_property}_sample_se.csv")
se_df.to_csv(se_csv, index=False)

print(f"\nResults saved to: {output_dir}")


# ==============================================================
# 8. Plot: sample-level SE distribution (proxy for SEP comparison)
# ==============================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(6, 5))
data_to_plot = [se_plsr, se_svr, se_tab]
bp = ax.boxplot(data_to_plot, tick_labels=["PLSR", "SVR", "NIRSpecPFN"],
                whis=3.0, showfliers=False)
ax.set_ylabel("Sample-level Squared Error (SE)", fontsize=18)
ax.set_title("Sample-Level SE Distribution\n(SE ∝ SEP², fixed test set)", fontsize=15)
ax.tick_params(axis="both", which="major", labelsize=16)

# Add RMSE / SEP labels below each box
y_max = max(d.max() for d in data_to_plot)
y_min = min(d.min() for d in data_to_plot)
yr = max(y_max - y_min, 1e-6)

for i, m in enumerate(["PLSR", "SVR", "NIRSpecPFN"], 1):
    rmse = final_df[final_df["Model"] == m]["Test_RMSE"].values[0]
    sep = final_df[final_df["Model"] == m]["Test_SEP"].values[0]
    sig = sig_df[sig_df["Model"] == m]["Sig"].values[0] if m in sig_df["Model"].values else ""
    ax.text(i, y_min - 0.03 * yr,
            f"RMSE={rmse:.3f}\nSEP={sep:.3f}{sig}",
            ha="center", va="top", fontsize=10, color="navy")

# Significance bars
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
plot_path = os.path.join(output_dir, f"se_distribution_{target_property.split(' ')[0]}.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Plot saved to: {plot_path}")
print("\n>>> Done.")