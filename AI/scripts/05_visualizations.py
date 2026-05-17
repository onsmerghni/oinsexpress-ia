"""
GENERATION DE VISUALISATIONS — VERSION CORRIGEE
=============================
Cree les graphiques utiles pour le rapport :
- Matrice de confusion
- Top 20 features importance
- Distribution des classes
- Courbes ROC
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RPT_DIR   = ROOT / "reports"

if not (ROOT / "data").exists():
    import sys
    print(f"ERREUR : dossier 'data' introuvable.", file=sys.stderr)
    sys.exit(1)

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
model  = joblib.load(MODEL_DIR / "xgb_driver_behavior.pkl")
le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
class_names = list(le.classes_)
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# 1) Matrice de confusion
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Matrice de confusion (effectifs)")
axes[0].set_ylabel("Vraie classe"); axes[0].set_xlabel("Prediction")

sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Matrice de confusion (normalisee par ligne)")
axes[1].set_ylabel("Vraie classe"); axes[1].set_xlabel("Prediction")
plt.tight_layout()
plt.savefig(RPT_DIR / "confusion_matrix.png", dpi=120)
plt.close()
print("OK confusion_matrix.png")

# ----------------------------------------------------------------------
# 2) Feature importance (Top 20) — VERSION CORRIGEE
# ----------------------------------------------------------------------
# Le modele est maintenant un XGBClassifier natif, pas un CalibratedClassifierCV.
# On accede directement a feature_importances_.
# Fallback : on lit le CSV genere par 03_train_xgboost.py.

try:
    # Cas 1 : le modele est un XGBClassifier natif (cas actuel)
    importances_array = model.feature_importances_
    print("      (importances lues directement depuis le modele XGBoost)")
except AttributeError:
    try:
        # Cas 2 : le modele est un CalibratedClassifierCV (au cas ou on revient en arriere)
        clf = model.calibrated_classifiers_[0]
        base_model = getattr(clf, "estimator", None) or getattr(clf, "base_estimator", None)
        importances_array = base_model.feature_importances_
        print("      (importances lues depuis le modele calibre)")
    except Exception:
        # Cas 3 : on lit le CSV deja genere par 03_train_xgboost.py
        imp_csv = pd.read_csv(RPT_DIR / "feature_importances.csv")
        # Reordonner selon feature_cols
        imp_dict = dict(zip(imp_csv["feature"], imp_csv["importance"]))
        importances_array = np.array([imp_dict.get(f, 0.0) for f in feature_cols])
        print("      (importances lues depuis le CSV reports/feature_importances.csv)")

imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances_array,
}).sort_values("importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=imp, y="feature", x="importance",
            hue="feature", palette="viridis", legend=False, ax=ax)
ax.set_title("Top 20 features les plus importantes (XGBoost)")
ax.set_xlabel("Importance (gain)"); ax.set_ylabel("")
plt.tight_layout()
plt.savefig(RPT_DIR / "feature_importance.png", dpi=120)
plt.close()
print("OK feature_importance.png")

# ----------------------------------------------------------------------
# 3) Courbes ROC
# ----------------------------------------------------------------------
y_score = model.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
n_classes = len(class_names)

if n_classes == 2:
    pos_idx = list(class_names).index("AGGRESSIVE") if "AGGRESSIVE" in class_names else 1
    fpr, tpr, _ = roc_curve(y_test == pos_idx, y_score[:, pos_idx])
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"AGGRESSIVE vs NORMAL (AUC = {auc(fpr, tpr):.3f})")
else:
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, cn in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=colors[i % 3], lw=2,
                label=f"{cn} (AUC = {auc(fpr, tpr):.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatoire")
ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
ax.set_title("Courbe ROC" + (" (binaire)" if n_classes == 2 else " par classe"))
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RPT_DIR / "roc_curves.png", dpi=120)
plt.close()
print("OK roc_curves.png")

# ----------------------------------------------------------------------
# 4) Distribution des classes
# ----------------------------------------------------------------------
y_train = np.load(DATA_DIR / "y_train.npy")
y_all = np.concatenate([y_train, y_test])

train_counts = pd.Series(y_train).map(dict(enumerate(class_names))).value_counts()
test_counts  = pd.Series(y_test).map(dict(enumerate(class_names))).value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
df_dist = pd.DataFrame({"Train": train_counts, "Test": test_counts}).reindex(class_names)
df_dist.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"])
ax.set_title("Distribution des classes (Train vs Test)")
ax.set_ylabel("Nombre d'echantillons"); ax.set_xlabel("")
plt.xticks(rotation=0)
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(RPT_DIR / "class_distribution.png", dpi=120)
plt.close()
print("OK class_distribution.png")

# ----------------------------------------------------------------------
# 5) Distribution des features cles par classe
# ----------------------------------------------------------------------
df_raw = pd.read_csv(DATA_DIR / "sensor_raw.csv").rename(columns={"Target(Class)": "Target"})
LABEL_MAP = {1: "NORMAL", 2: "NORMAL", 3: "AGGRESSIVE", 4: "AGGRESSIVE"}
df_raw["Behavior"] = df_raw["Target"].map(LABEL_MAP)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
features_to_plot = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
for ax, f in zip(axes.flatten(), features_to_plot):
    sns.boxplot(data=df_raw, x="Behavior", y=f, ax=ax,
                hue="Behavior", palette="Set2", legend=False,
                order=class_names)
    ax.set_title(f"Distribution de {f} par classe")
    ax.set_xlabel("")
plt.tight_layout()
plt.savefig(RPT_DIR / "raw_signals_by_class.png", dpi=120)
plt.close()
print("OK raw_signals_by_class.png")

print("\nTous les graphiques generes dans :", RPT_DIR)