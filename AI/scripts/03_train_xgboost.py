"""
ENTRAINEMENT XGBOOST — VERSION REGULARISATION FORTE
=====================================================================
Modifications :
  - Grille TRES restreinte avec valeurs FORTEMENT regularisees
  - Objectif : Train ~0.92, Test ~0.87, Gap < 0.10
  - Le tuning ne peut plus choisir des valeurs "puissantes"
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import (
    GroupKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RPT_DIR   = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

if not (ROOT / "data").exists():
    import sys
    print(f"ERREUR : dossier 'data' introuvable.", file=sys.stderr)
    sys.exit(1)

RNG = 42

# 1) Chargement
print("[1/5] Chargement...", flush=True)
X_train = np.load(DATA_DIR / "X_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")
groups_train = np.load(DATA_DIR / "groups_train.npy")
le      = joblib.load(MODEL_DIR / "label_encoder.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
class_names = list(le.classes_)
print(f"      Train: {X_train.shape}  Test: {X_test.shape}  Classes: {class_names}", flush=True)
print(f"      Encoding: {dict(zip(class_names, range(len(class_names))))}")
print(f"      Nombre de groupes en train : {len(np.unique(groups_train))}", flush=True)

sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

# 2) Baseline
print("\n[2/5] Baseline XGBoost regularise (GroupKFold 5-fold)...", flush=True)
t0 = time.perf_counter()
n_classes = len(class_names)
XGB_OBJECTIVE = "binary:logistic"
XGB_EVAL_METRIC = "logloss"

baseline = xgb.XGBClassifier(
    n_estimators=80,
    max_depth=2,
    learning_rate=0.05,
    gamma=1.0,
    min_child_weight=8,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    objective=XGB_OBJECTIVE, eval_metric=XGB_EVAL_METRIC,
    random_state=RNG, n_jobs=-1, tree_method="hist",
)

gkf5 = GroupKFold(n_splits=5)
baseline_scores = cross_val_score(
    baseline, X_train, y_train,
    groups=groups_train, cv=gkf5, scoring="f1_macro", n_jobs=1,
    fit_params={"sample_weight": sample_w},
)
print(f"      CV F1-macro: {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}", flush=True)

# 3) Tuning AVEC GRILLE TRES REGULARISEE
print("\n[3/5] Tuning rapide (grille FORTEMENT regularisee, 10 essais x GroupKFold 3-fold)...", flush=True)
t0 = time.perf_counter()

# === GRILLE TRES STRICTE — toutes les valeurs sont regularisees ===
param_dist = {
    "n_estimators":      [50, 80, 100, 120],          # MAX 120 (avant: 200-400)
    "max_depth":         [2, 3],                       # arbres simples
    "learning_rate":     [0.03, 0.05, 0.08],           # lent
    "subsample":         [0.6, 0.7, 0.8],              # tres diversifie
    "colsample_bytree":  [0.5, 0.7],                   # tres diversifie
    "min_child_weight":  [5, 7, 10],                   # MINIMUM 5 (avant: 1)
    "gamma":             [0.5, 1.0, 1.5],              # MINIMUM 0.5 (avant: 0)
    "reg_alpha":         [0.3, 0.5, 1.0],              # forte L1
    "reg_lambda":        [2.0, 3.0, 5.0],              # forte L2
}

gkf3 = GroupKFold(n_splits=3)
search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(
        objective=XGB_OBJECTIVE, eval_metric=XGB_EVAL_METRIC,
        random_state=RNG, n_jobs=1, tree_method="hist",
    ),
    param_distributions=param_dist,
    n_iter=15, scoring="f1_macro", cv=gkf3,
    random_state=RNG, n_jobs=2, verbose=1,
)
search.fit(X_train, y_train, groups=groups_train, sample_weight=sample_w)
best = search.best_estimator_
print(f"      Meilleur F1-macro CV: {search.best_score_:.4f}  ({time.perf_counter()-t0:.2f}s)", flush=True)
print(f"      Meilleurs hyperparams (regularises forts):")
for k, v in search.best_params_.items():
    print(f"        {k}: {v}")

print("\n[3 bis] Pas de calibration explicite")
final_model = best

# ============================================================
# 4) EVALUATION TRAIN + TEST
# ============================================================
print("\n[4/5] Evaluation sur Train ET Test...", flush=True)

# === TRAIN ===
y_train_pred  = final_model.predict(X_train)
y_train_proba = final_model.predict_proba(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
train_f1m = f1_score(y_train, y_train_pred, average="macro")
try:
    train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
except Exception:
    train_auc = float("nan")

# === TEST ===
y_pred  = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
f1w = f1_score(y_test, y_pred, average="weighted")
try:
    auc = roc_auc_score(y_test, y_proba[:, 1])
except Exception:
    auc = float("nan")

# === TEMPS D'INFERENCE ===
N_REPEATS = 100
t0 = time.perf_counter()
for _ in range(N_REPEATS):
    final_model.predict(X_test)
total_pred_time = time.perf_counter() - t0
pred_time_ms_batch = (total_pred_time / N_REPEATS) * 1000
pred_time_us_per_sample = (pred_time_ms_batch * 1000) / len(X_test)

# === COMPARATIF TRAIN vs TEST ===
print("\n      ============================================================")
print("      COMPARAISON TRAIN vs TEST")
print("      ============================================================")
print(f"      {'Metrique':<15} {'Train':<12} {'Test':<12} {'Ecart':<10}")
print(f"      {'-'*49}")
print(f"      {'Accuracy':<15} {train_acc:<12.4f} {acc:<12.4f} {abs(train_acc-acc):<10.4f}")
print(f"      {'F1-macro':<15} {train_f1m:<12.4f} {f1m:<12.4f} {abs(train_f1m-f1m):<10.4f}")
print(f"      {'AUC':<15} {train_auc:<12.4f} {auc:<12.4f} {abs(train_auc-auc):<10.4f}")
print()

gap_train_test = train_acc - acc
print(f"      Ecart Accuracy Train - Test = {gap_train_test:.4f}")
if gap_train_test > 0.15:
    print(f"      ATTENTION : ecart > 0.15 => OVERFITTING SUSPECT")
elif gap_train_test > 0.05:
    print(f"      OK : ecart modere (< 0.15), generalisation correcte")
else:
    print(f"      EXCELLENT : ecart < 0.05, excellente generalisation")

gap_cv_test = abs(search.best_score_ - f1m)
print(f"\n      Ecart |CV - Test| = {gap_cv_test:.4f}")
if gap_cv_test < 0.05:
    print(f"      EXCELLENT : evaluation rigoureuse (ecart < 0.05)")
elif gap_cv_test < 0.10:
    print(f"      OK : ecart acceptable (< 0.10)")
else:
    print(f"      Acceptable : ecart modere")

print(f"\n      Temps d'inference (sur {N_REPEATS} repetitions) :")
print(f"        - Batch ({len(X_test)} echantillons) : {pred_time_ms_batch:.4f} ms")
print(f"        - Par echantillon                   : {pred_time_us_per_sample:.2f} us")

# === DETAIL TEST ===
print("\n      ============================================================")
print("      DETAIL DU TEST SET")
print("      ============================================================")
print(f"      Accuracy : {acc:.4f}")
print(f"      F1-macro : {f1m:.4f}")
print(f"      F1-weight: {f1w:.4f}")
print(f"      AUC      : {auc:.4f}")
print("\n      Classification report (test):")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("      Confusion matrix test:")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

print("\n      Confusion matrix train:")
cm_train = confusion_matrix(y_train, y_train_pred)
print(pd.DataFrame(cm_train, index=class_names, columns=class_names))

print("\n      Distribution des probabilites P(AGGRESSIVE) sur le test :")
idx_aggr = class_names.index("AGGRESSIVE")
proba_aggr = y_proba[:, idx_aggr]
print(f"        min={proba_aggr.min():.3f}  q25={np.percentile(proba_aggr,25):.3f}  "
      f"median={np.median(proba_aggr):.3f}  q75={np.percentile(proba_aggr,75):.3f}  "
      f"max={proba_aggr.max():.3f}")

# 5) Importances + save
print("\n[5/5] Top 15 features les plus importantes:")
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False)
print(importances.head(15).to_string(index=False))

joblib.dump(final_model, MODEL_DIR / "xgb_driver_behavior.pkl")
final_model.save_model(str(MODEL_DIR / "xgb_driver_behavior.json"))
importances.to_csv(RPT_DIR / "feature_importances.csv", index=False)

report = {
    "train_accuracy":   float(train_acc),
    "train_f1_macro":   float(train_f1m),
    "train_auc":        float(train_auc),
    "test_accuracy":    float(acc),
    "test_f1_macro":    float(f1m),
    "test_f1_weighted": float(f1w),
    "test_auc":         float(auc),
    "gap_train_test_accuracy": float(gap_train_test),
    "cv_f1_macro_mean_baseline": float(baseline_scores.mean()),
    "cv_f1_macro_std_baseline":  float(baseline_scores.std()),
    "cv_f1_macro_best_tuned":    float(search.best_score_),
    "cv_method":        "GroupKFold + sample_weight=balanced + STRONG regularization",
    "best_params":      search.best_params_,
    "regularization_strategy": "max_depth<=3, gamma>=0.5, min_child_weight>=5, reg_alpha>=0.3, reg_lambda>=2.0",
    "classes":          class_names,
    "n_features":       len(feature_cols),
    "n_train":          int(len(X_train)),
    "n_test":           int(len(X_test)),
    "n_groups_train":   int(len(np.unique(groups_train))),
    "confusion_matrix_test":  cm.tolist(),
    "confusion_matrix_train": cm_train.tolist(),
    "pred_time_ms_batch":      float(pred_time_ms_batch),
    "pred_time_us_per_sample": float(pred_time_us_per_sample),
    "proba_distribution_test": {
        "min": float(proba_aggr.min()),
        "q25": float(np.percentile(proba_aggr, 25)),
        "median": float(np.median(proba_aggr)),
        "q75": float(np.percentile(proba_aggr, 75)),
        "max": float(proba_aggr.max()),
    },
}
with open(RPT_DIR / "training_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n      OK. Modele dans: {MODEL_DIR / 'xgb_driver_behavior.pkl'}")