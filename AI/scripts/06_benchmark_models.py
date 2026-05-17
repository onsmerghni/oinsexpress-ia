"""
BENCHMARK : Comparaison de plusieurs modeles
=============================================
MISE A JOUR : XGBoost utilise les memes hyperparametres regularises
que le modele final (script 03) pour une coherence parfaite.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RPT_DIR   = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

if not DATA_DIR.exists():
    import sys
    print(f"ERREUR : dossier 'data' introuvable a {DATA_DIR}", file=sys.stderr)
    sys.exit(1)

RNG = 42
N_REPEATS_PRED = 100

# -----------------------------------------------------------------------------
# Chargement
# -----------------------------------------------------------------------------
print("=" * 70)
print("BENCHMARK : 7 modeles compares dans les memes conditions")
print("=" * 70)
print("\n[1/4] Chargement train/test...")
X_train = np.load(DATA_DIR / "X_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")
le      = joblib.load(MODEL_DIR / "label_encoder.pkl")
class_names = list(le.classes_)
print(f"      Train: {X_train.shape}  Test: {X_test.shape}")
print(f"      Classes: {class_names}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# Modeles
# -----------------------------------------------------------------------------
print("\n[2/4] Definition des 7 modeles...")
print("      NOTE: XGBoost et LightGBM utilisent les hyperparams REGULARISES")
print("            (memes que le modele final, script 03)")

models = [
    ("Logistic Regression",
     LogisticRegression(max_iter=2000, random_state=RNG, n_jobs=-1),
     True),
    ("K-Nearest Neighbors",
     KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
     True),
    ("SVM (RBF kernel)",
     SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RNG),
     True),

    # Random Forest avec regularisation (max_depth limite)
    ("Random Forest",
     RandomForestClassifier(
        n_estimators=100,        # reduit (etait 300)
        max_depth=5,             # AJOUT regularisation (etait None)
        min_samples_leaf=3,      # AJOUT regularisation
        max_features="sqrt",     # diversite
        random_state=RNG, n_jobs=-1),
     False),

    # XGBoost REGULARISE (memes hyperparams que le modele final)
    ("XGBoost",
     xgb.XGBClassifier(
        n_estimators=50,         # tres reduit (etait 200)
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1.5,               # FORTE regularisation
        min_child_weight=5,      # FORTE regularisation
        reg_alpha=1.0,           # AJOUT L1
        reg_lambda=3.0,          # AJOUT L2
        objective="binary:logistic", eval_metric="logloss",
        random_state=RNG, n_jobs=-1, tree_method="hist",
     ),
     False),

    # LightGBM REGULARISE (coherent avec XGBoost)
    ("LightGBM",
     lgb.LGBMClassifier(
        n_estimators=50,         # tres reduit
        max_depth=3,
        learning_rate=0.08,
        num_leaves=8,            # reduit (etait 31)
        min_child_samples=10,    # AJOUT regularisation
        reg_alpha=1.0,           # AJOUT L1
        reg_lambda=3.0,          # AJOUT L2
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RNG, n_jobs=-1, verbose=-1,
     ),
     False),

    ("MLP (neural net)",
     MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=300, early_stopping=True, random_state=RNG,
     ),
     True),
]

# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
print("\n[3/4] Entrainement et evaluation des 7 modeles...")
print(f"      (Temps de prediction moyenne sur {N_REPEATS_PRED} repetitions)\n")

results = []
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

for name, model, use_scaled in models:
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    # CV
    t0 = time.perf_counter()
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv5,
                                scoring="f1_macro", n_jobs=1)
    cv_time = time.perf_counter() - t0

    # Entrainement
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    train_time = time.perf_counter() - t0

    # Train Acc (pour overfitting)
    y_train_pred = model.predict(Xtr)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")

    # Test avec timing precis
    t0 = time.perf_counter()
    for _ in range(N_REPEATS_PRED):
        y_pred = model.predict(Xte)
    total_pred_time = time.perf_counter() - t0
    pred_time_ms = (total_pred_time / N_REPEATS_PRED) * 1000
    pred_time_us_per_sample = (pred_time_ms * 1000) / len(Xte)

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(Xte)
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")
    cm  = confusion_matrix(y_test, y_pred)
    gap = train_acc - acc

    results.append({
        "model":         name,
        "cv_f1_mean":    float(cv_scores.mean()),
        "cv_f1_std":     float(cv_scores.std()),
        "train_accuracy": float(train_acc),
        "train_f1_macro": float(train_f1),
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1m),
        "test_f1_weight": float(f1w),
        "test_auc":      float(auc),
        "gap_train_test": float(gap),
        "train_time_s":  float(train_time),
        "pred_time_ms":  float(pred_time_ms),
        "pred_time_us_per_sample": float(pred_time_us_per_sample),
        "confusion_matrix": cm.tolist(),
    })

    print(f"  {name:25s}  CV-F1={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}  "
          f"TrainAcc={train_acc:.4f}  TestAcc={acc:.4f}  Gap={gap:+.4f}  "
          f"AUC={auc:.4f}  Train={train_time:.3f}s  Pred={pred_time_ms:.4f}ms")

# -----------------------------------------------------------------------------
# Synthese
# -----------------------------------------------------------------------------
print("\n[4/4] Synthese du benchmark\n")

df = pd.DataFrame(results)
df_sorted = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

print("=" * 130)
print(f"{'Rang':<5} {'Modele':<22} {'CV F1':<9} {'Train Acc':<11} {'Test Acc':<10} {'Gap':<9} {'F1':<9} {'AUC':<9} {'Train(s)':<10} {'Pred(ms)':<10}")
print("=" * 130)
for i, row in df_sorted.iterrows():
    gap_str = f"{row['gap_train_test']:+.4f}"
    print(f"#{i+1:<4} {row['model']:<22} "
          f"{row['cv_f1_mean']:.4f}    "
          f"{row['train_accuracy']:.4f}     "
          f"{row['test_accuracy']:.4f}    "
          f"{gap_str:<9} "
          f"{row['test_f1_macro']:.4f}    "
          f"{row['test_auc']:.4f}    "
          f"{row['train_time_s']:<10.3f} "
          f"{row['pred_time_ms']:<10.4f}")
print("=" * 130)

print("\n  Legende :")
print("    - Gap = Train Accuracy - Test Accuracy")
print("    - Gap < 0.05  : excellente generalisation")
print("    - Gap < 0.15  : generalisation acceptable")
print("    - Gap >= 0.15 : overfitting probable")
print(f"    - Pred (ms)   : temps batch test ({len(X_test)} echantillons), moyenne sur {N_REPEATS_PRED} repetitions")
print("    - NOTE        : XGBoost, LightGBM et Random Forest sont REGULARISES")

df.to_csv(RPT_DIR / "benchmark_results.csv", index=False)
with open(RPT_DIR / "benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------------------------------------------------------
# Visualisations
# -----------------------------------------------------------------------------
sns.set_style("whitegrid")

# 1. Bar chart Train vs Test
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
df_plot = df_sorted.copy()
x_pos = np.arange(len(df_plot))
width = 0.35
ax.barh(x_pos - width/2, df_plot["train_accuracy"], width,
        label="Train Accuracy", color="#3498db", edgecolor="black")
ax.barh(x_pos + width/2, df_plot["test_accuracy"], width,
        label="Test Accuracy", color="#e74c3c", edgecolor="black")
ax.set_yticks(x_pos)
ax.set_yticklabels(df_plot["model"])
ax.set_xlabel("Accuracy")
ax.set_title("Train vs Test Accuracy par modele\n(detection overfitting)")
ax.set_xlim(0.5, 1.05)
ax.invert_yaxis()
ax.legend(loc="lower right")
for i, row in df_plot.iterrows():
    ax.text(row["train_accuracy"] + 0.005, i - width/2,
            f"{row['train_accuracy']:.3f}", va="center", fontsize=8)
    ax.text(row["test_accuracy"] + 0.005, i + width/2,
            f"{row['test_accuracy']:.3f}", va="center", fontsize=8)

# 2. Scatter F1 vs Train time
ax = axes[1]
colors = sns.color_palette("viridis", len(df_plot))
for i, row in df_plot.iterrows():
    ax.scatter(row["train_time_s"], row["test_f1_macro"], s=200,
               color=colors[i], label=row["model"], edgecolor="black")
    ax.annotate(row["model"],
                (row["train_time_s"], row["test_f1_macro"]),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Temps d'entrainement (secondes, echelle log)")
ax.set_ylabel("F1-macro (test set)")
ax.set_title("Performance vs Temps d'entrainement")
ax.set_xscale("log")

plt.tight_layout()
plt.savefig(RPT_DIR / "benchmark_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  OK benchmark_comparison.png")

# 3. Matrices de confusion
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, row in df_sorted.iterrows():
    if i >= 8:
        break
    cm = np.array(row["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[i], cbar=False)
    axes[i].set_title(f"#{i+1}. {row['model']}\nAcc={row['test_accuracy']:.4f}")
    axes[i].set_xlabel("Prediction"); axes[i].set_ylabel("Vraie classe")
for j in range(len(df_sorted), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig(RPT_DIR / "benchmark_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  OK benchmark_confusion_matrices.png")

# 4. Heatmap metriques
fig, ax = plt.subplots(figsize=(13, 5))
metrics_df = df_sorted[["model", "cv_f1_mean", "train_accuracy", "test_accuracy",
                        "test_f1_macro", "test_auc"]].set_index("model")
metrics_df.columns = ["CV F1", "Train Acc", "Test Acc", "F1-macro", "AUC"]
sns.heatmap(metrics_df, annot=True, fmt=".4f", cmap="RdYlGn",
            vmin=0.7, vmax=1.0, ax=ax, cbar_kws={"label": "Score"})
ax.set_title("Heatmap des performances (Train vs Test)")
ax.set_ylabel(""); ax.set_xlabel("")
plt.tight_layout()
plt.savefig(RPT_DIR / "benchmark_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  OK benchmark_heatmap.png")

# -----------------------------------------------------------------------------
# Conclusion
# -----------------------------------------------------------------------------
best = df_sorted.iloc[0]
print("\n" + "=" * 70)
print("CONCLUSION DU BENCHMARK")
print("=" * 70)
print(f"\nMeilleur modele : {best['model']}")
print(f"  - F1-macro test : {best['test_f1_macro']:.4f}")
print(f"  - Train Acc     : {best['train_accuracy']:.4f}")
print(f"  - Test Acc      : {best['test_accuracy']:.4f}")
print(f"  - Gap Train-Test: {best['gap_train_test']:+.4f}")
print(f"  - AUC test      : {best['test_auc']:.4f}")
print(f"  - Temps train   : {best['train_time_s']:.3f}s")
print(f"  - Temps pred    : {best['pred_time_ms']:.4f}ms (batch de {len(X_test)} echantillons)")

# Trouver XGBoost dans le ranking
xgb_row = df_sorted[df_sorted["model"] == "XGBoost"].iloc[0]
xgb_rank = df_sorted[df_sorted["model"] == "XGBoost"].index[0] + 1
print(f"\nXGBoost (modele final retenu) :")
print(f"  - Rang          : #{xgb_rank}")
print(f"  - F1-macro test : {xgb_row['test_f1_macro']:.4f}")
print(f"  - Train Acc     : {xgb_row['train_accuracy']:.4f}")
print(f"  - Test Acc      : {xgb_row['test_accuracy']:.4f}")
print(f"  - Gap Train-Test: {xgb_row['gap_train_test']:+.4f}")
print(f"  - AUC test      : {xgb_row['test_auc']:.4f}")

print(f"\nFichiers generes dans {RPT_DIR.name}/ :")
print("  - benchmark_results.csv")
print("  - benchmark_results.json")
print("  - benchmark_comparison.png")
print("  - benchmark_confusion_matrices.png")
print("  - benchmark_heatmap.png")