"""
EVALUATION FINALE DU MODELE DE PRODUCTION (XGBoost)
====================================================
Charge le modele final sauvegarde (xgb_driver_behavior.pkl) et produit
un rapport complet "production-ready" qui cloture la partie IA :

  - Metriques officielles sur le test hold-out
  - Matrice de confusion + classification report
  - Calibration des seuils WARNING/CRITICAL
  - Analyse des erreurs (faux positifs / faux negatifs)
  - Specifications du modele pour le deploiement
  - Validation des artefacts (.pkl, scaler, encoder)

Sortie :
  - reports/final_evaluation_report.json
  - reports/final_evaluation_summary.md
  - reports/final_threshold_calibration.png
  - reports/final_confusion_matrix.png
  - reports/final_error_analysis.csv
"""

import json
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Convertit les types NumPy en types Python natifs pour la serialisation JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RPT_DIR   = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("EVALUATION FINALE DU MODELE DE PRODUCTION (XGBoost)")
print("Cloture de la partie IA - rapport production-ready")
print("=" * 70)

# -----------------------------------------------------------------------------
# 1) Chargement des artefacts
# -----------------------------------------------------------------------------
print("\n[1/6] Chargement des artefacts...")

required_files = [
    MODEL_DIR / "xgb_driver_behavior.pkl",
    MODEL_DIR / "label_encoder.pkl",
    MODEL_DIR / "scaler.pkl",
    MODEL_DIR / "feature_cols.pkl",
    DATA_DIR  / "X_test.npy",
    DATA_DIR  / "y_test.npy",
    DATA_DIR  / "X_train.npy",
    DATA_DIR  / "y_train.npy",
]
missing = [p for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError(
        f"Artefacts manquants : {missing}\n"
        f"Lancez d'abord 02_preprocess.py puis 03_train_xgboost.py"
    )

model        = joblib.load(MODEL_DIR / "xgb_driver_behavior.pkl")
le           = joblib.load(MODEL_DIR / "label_encoder.pkl")
scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

class_names = list(le.classes_)
idx_aggr = class_names.index("AGGRESSIVE")
print(f"      Modele charge        : {type(model).__name__}")
print(f"      Classes              : {class_names}")
print(f"      Index AGGRESSIVE     : {idx_aggr}")
print(f"      Nombre de features   : {len(feature_cols)}")
print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

# -----------------------------------------------------------------------------
# 2) Predictions
# -----------------------------------------------------------------------------
print("\n[2/6] Predictions sur le test hold-out...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
p_aggr  = y_proba[:, idx_aggr]

# Train (pour gap)
y_train_pred = model.predict(X_train)

# -----------------------------------------------------------------------------
# 3) Metriques officielles
# -----------------------------------------------------------------------------
print("\n[3/6] Calcul des metriques officielles...")

acc       = accuracy_score(y_test, y_pred)
f1_macro  = f1_score(y_test, y_pred, average="macro")
f1_weight = f1_score(y_test, y_pred, average="weighted")
prec_macro= precision_score(y_test, y_pred, average="macro")
rec_macro = recall_score(y_test, y_pred, average="macro")
# AUC : on utilise (y_test == idx_aggr) pour avoir un label binaire "vrai AGGRESSIVE"
# aligne avec la proba p_aggr = P(AGGRESSIVE). Sinon, si idx_aggr=0, on aurait
# une AUC inversee (proche de 0 au lieu de proche de 1).
y_test_is_aggr = (y_test == idx_aggr).astype(int)
auc       = roc_auc_score(y_test_is_aggr, p_aggr)
ap        = average_precision_score(y_test_is_aggr, p_aggr)

train_acc = accuracy_score(y_train, y_train_pred)
gap = train_acc - acc

# Per-class (binaire)
prec_per = precision_score(y_test, y_pred, average=None)
rec_per  = recall_score(y_test, y_pred, average=None)
f1_per   = f1_score(y_test, y_pred, average=None)

cm = confusion_matrix(y_test, y_pred)

print(f"      Accuracy             : {acc:.4f}")
print(f"      F1-macro             : {f1_macro:.4f}")
print(f"      F1-weighted          : {f1_weight:.4f}")
print(f"      Precision macro      : {prec_macro:.4f}")
print(f"      Recall macro         : {rec_macro:.4f}")
print(f"      AUC (probabiliste)   : {auc:.4f}")
print(f"      Average Precision    : {ap:.4f}")
print(f"      Gap train-test       : {gap:+.4f}")

print(f"\n      Par classe :")
for i, cn in enumerate(class_names):
    print(f"        {cn:<12} -> Precision={prec_per[i]:.4f}  Recall={rec_per[i]:.4f}  F1={f1_per[i]:.4f}")

print(f"\n      Matrice de confusion :")
print(pd.DataFrame(cm, index=class_names, columns=class_names))

# -----------------------------------------------------------------------------
# 4) Calibration des seuils d'alerte
# -----------------------------------------------------------------------------
print("\n[4/6] Calibration des seuils WARNING / CRITICAL...")

# Charger seuils existants si presents, sinon valeurs par defaut
SEUIL_WARNING  = 0.50
SEUIL_CRITICAL = 0.95
thr_path = MODEL_DIR / "alert_thresholds.pkl"
if thr_path.exists():
    try:
        thr = joblib.load(thr_path)
        SEUIL_WARNING  = thr.get("warning", SEUIL_WARNING)
        SEUIL_CRITICAL = thr.get("critical", SEUIL_CRITICAL)
        print(f"      Seuils charges depuis alert_thresholds.pkl")
    except Exception:
        print(f"      alert_thresholds.pkl illisible, valeurs par defaut.")

print(f"      WARNING  >= {SEUIL_WARNING:.2f}  -> declenche alerte de niveau 1")
print(f"      CRITICAL >= {SEUIL_CRITICAL:.2f}  -> declenche alerte de niveau 2 (responsable)")

# Distribution des probas par classe reelle
mask_aggr = (y_test == idx_aggr)
proba_when_aggr   = p_aggr[mask_aggr]
proba_when_normal = p_aggr[~mask_aggr]

n_warning  = int((p_aggr >= SEUIL_WARNING).sum())
n_critical = int((p_aggr >= SEUIL_CRITICAL).sum())
n_normal_alert = int(n_warning - n_critical)

# Detail des alertes
tp_warning = int(((p_aggr >= SEUIL_WARNING) & mask_aggr).sum())
fp_warning = int(((p_aggr >= SEUIL_WARNING) & ~mask_aggr).sum())
tp_critical= int(((p_aggr >= SEUIL_CRITICAL) & mask_aggr).sum())
fp_critical= int(((p_aggr >= SEUIL_CRITICAL) & ~mask_aggr).sum())

print(f"\n      Repartition des alertes sur le test :")
print(f"        Total declenchees       : {n_warning} sur {len(y_test)}")
print(f"        - dont WARNING (50-94%) : {n_normal_alert}")
print(f"        - dont CRITICAL (>=95%) : {n_critical}")
print(f"        Vrais positifs WARNING  : {tp_warning} / {n_warning} ({100*tp_warning/max(n_warning,1):.1f}%)")
print(f"        Vrais positifs CRITICAL : {tp_critical} / {n_critical} ({100*tp_critical/max(n_critical,1):.1f}%)")

# -----------------------------------------------------------------------------
# 5) Analyse des erreurs
# -----------------------------------------------------------------------------
print("\n[5/6] Analyse des erreurs...")

errors_idx = np.where(y_pred != y_test)[0]
errors_df = pd.DataFrame({
    "index_test": errors_idx,
    "vraie_classe":   [class_names[c] for c in y_test[errors_idx]],
    "predite_classe": [class_names[c] for c in y_pred[errors_idx]],
    "P_AGGRESSIVE":   p_aggr[errors_idx],
    "P_NORMAL":       1 - p_aggr[errors_idx],
})
errors_df["type_erreur"] = errors_df.apply(
    lambda r: "FAUX_POSITIF" if r["vraie_classe"] == "NORMAL" else "FAUX_NEGATIF",
    axis=1,
)
errors_df = errors_df.sort_values("P_AGGRESSIVE", ascending=False).reset_index(drop=True)

n_fp = int((errors_df["type_erreur"] == "FAUX_POSITIF").sum())
n_fn = int((errors_df["type_erreur"] == "FAUX_NEGATIF").sum())

print(f"      Total erreurs        : {len(errors_idx)} / {len(y_test)}")
print(f"      - Faux positifs (NORMAL classe AGGRESSIVE) : {n_fp}")
print(f"      - Faux negatifs (AGGRESSIVE classe NORMAL) : {n_fn}  <- pire cas")

if len(errors_df) > 0:
    print(f"\n      Detail des erreurs (triees par P(AGGRESSIVE) decroissante) :")
    print(errors_df.to_string(index=False))

errors_df.to_csv(RPT_DIR / "final_error_analysis.csv", index=False)
print(f"\n      OK final_error_analysis.csv")

# -----------------------------------------------------------------------------
# 6) Visualisations
# -----------------------------------------------------------------------------
print("\n[6/6] Generation des visualisations...")
sns.set_style("whitegrid")

# 6.1 Matrice de confusion finale
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Prediction", fontweight="bold")
ax.set_ylabel("Vraie classe", fontweight="bold")
ax.set_title(f"Matrice de confusion - Modele final XGBoost\n"
             f"Accuracy={acc:.4f} | F1-macro={f1_macro:.4f} | AUC={auc:.4f}",
             fontweight="bold")
plt.tight_layout()
plt.savefig(RPT_DIR / "final_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"      OK final_confusion_matrix.png")

# 6.2 Calibration des seuils
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1 : distribution des probas par classe reelle + seuils
ax = axes[0]
bins = np.linspace(0, 1, 21)
ax.hist(proba_when_normal, bins=bins, alpha=0.6, label="Vrai NORMAL",
        color="#3498db", edgecolor="black")
ax.hist(proba_when_aggr, bins=bins, alpha=0.6, label="Vrai AGGRESSIVE",
        color="#e74c3c", edgecolor="black")
ax.axvline(SEUIL_WARNING, color="orange", linestyle="--", linewidth=2,
           label=f"WARNING ({SEUIL_WARNING:.2f})")
ax.axvline(SEUIL_CRITICAL, color="darkred", linestyle="--", linewidth=2,
           label=f"CRITICAL ({SEUIL_CRITICAL:.2f})")
ax.set_xlabel("P(AGGRESSIVE)")
ax.set_ylabel("Nombre d'echantillons test")
ax.set_title("Distribution des probabilites par classe reelle\nFondement des seuils d'alerte", fontweight="bold")
ax.legend(loc="upper center")

# Plot 2 : courbe ROC + seuils marques
ax = axes[1]
fpr, tpr, thresholds = roc_curve(y_test_is_aggr, p_aggr)
ax.plot(fpr, tpr, linewidth=2.5, color="#27ae60", label=f"ROC (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatoire")

# Trouver le point le plus proche du seuil
for seuil, label, color in [(SEUIL_WARNING, "WARNING", "orange"),
                              (SEUIL_CRITICAL, "CRITICAL", "darkred")]:
    idx = np.argmin(np.abs(thresholds - seuil))
    ax.scatter(fpr[idx], tpr[idx], s=150, color=color, edgecolor="black",
               zorder=5, label=f"{label} (seuil={seuil:.2f})")
    ax.annotate(f"  TPR={tpr[idx]:.2f}\n  FPR={fpr[idx]:.2f}",
                (fpr[idx], tpr[idx]),
                xytext=(10, -10), textcoords="offset points", fontsize=9)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Courbe ROC + seuils d'alerte", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
plt.savefig(RPT_DIR / "final_threshold_calibration.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"      OK final_threshold_calibration.png")

# -----------------------------------------------------------------------------
# 7) Rapport JSON final
# -----------------------------------------------------------------------------
report = {
    "model_info": {
        "type": "XGBoost Classifier (binaire)",
        "objective": "binary:logistic",
        "n_features": len(feature_cols),
        "classes": class_names,
        "model_file": "models/xgb_driver_behavior.pkl",
        "scaler_file": "models/scaler.pkl",
        "label_encoder_file": "models/label_encoder.pkl",
        "feature_cols_file": "models/feature_cols.pkl",
        "thresholds_file": "models/alert_thresholds.pkl",
    },
    "dataset_info": {
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "class_distribution_test": {
            class_names[c]: int((y_test == c).sum()) for c in np.unique(y_test)
        },
    },
    "metrics": {
        "test_accuracy":    float(acc),
        "test_f1_macro":    float(f1_macro),
        "test_f1_weighted": float(f1_weight),
        "test_precision_macro": float(prec_macro),
        "test_recall_macro": float(rec_macro),
        "test_auc":          float(auc),
        "test_avg_precision": float(ap),
        "train_accuracy":    float(train_acc),
        "gap_train_test":    float(gap),
        "per_class": {
            class_names[i]: {
                "precision": float(prec_per[i]),
                "recall":    float(rec_per[i]),
                "f1":        float(f1_per[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
    },
    "alert_thresholds": {
        "warning":  float(SEUIL_WARNING),
        "critical": float(SEUIL_CRITICAL),
        "test_alerts": {
            "n_warning":  n_warning,
            "n_critical": n_critical,
            "tp_warning":  tp_warning,
            "fp_warning":  fp_warning,
            "tp_critical": tp_critical,
            "fp_critical": fp_critical,
            "precision_warning":  float(tp_warning / max(n_warning, 1)),
            "precision_critical": float(tp_critical / max(n_critical, 1)),
        },
    },
    "errors": {
        "total": int(len(errors_idx)),
        "false_positives": n_fp,
        "false_negatives": n_fn,
    },
    "deployment": {
        "service": "Flask (oinsexpress_ia/flask_service/anomaly_detector.py)",
        "endpoint_predict": "POST /predict",
        "endpoint_health":  "GET /health",
        "real_time": True,
    },
    "production_ready": (
        gap < 0.10 and acc >= 0.85 and auc >= 0.90
    ),
}

with open(RPT_DIR / "final_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
print(f"\n      OK final_evaluation_report.json")

# -----------------------------------------------------------------------------
# 8) Rapport Markdown final
# -----------------------------------------------------------------------------
md_lines = [
    "# Evaluation finale du modele de production\n",
    "Ce document cloture la partie IA du projet. Il reporte les metriques ",
    "officielles du modele XGBoost de production, evalue sur le test hold-out, ",
    "et detaille la calibration du systeme d'alerte.\n",
    "## 1. Identification du modele\n",
    "| Element | Valeur |",
    "|---|---|",
    f"| Type | {report['model_info']['type']} |",
    f"| Fichier | `models/xgb_driver_behavior.pkl` |",
    f"| Classes | {', '.join(class_names)} |",
    f"| Nombre de features | {len(feature_cols)} |",
    f"| Scaler | `models/scaler.pkl` |",
    f"| Label encoder | `models/label_encoder.pkl` |\n",
    "## 2. Metriques sur le test hold-out\n",
    "| Metrique | Valeur |",
    "|---|---|",
    f"| **Accuracy** | **{acc:.4f}** |",
    f"| **F1-macro** | **{f1_macro:.4f}** |",
    f"| F1-weighted | {f1_weight:.4f} |",
    f"| Precision macro | {prec_macro:.4f} |",
    f"| Recall macro | {rec_macro:.4f} |",
    f"| **AUC** | **{auc:.4f}** |",
    f"| Average Precision | {ap:.4f} |",
    f"| Train Accuracy | {train_acc:.4f} |",
    f"| **Gap Train-Test** | **{gap:+.4f}** (excellent si < 0.10) |\n",
    "### 2.1 Performance par classe\n",
    "| Classe | Precision | Recall | F1 |",
    "|---|---|---|---|",
]
for i, cn in enumerate(class_names):
    md_lines.append(f"| **{cn}** | {prec_per[i]:.4f} | {rec_per[i]:.4f} | {f1_per[i]:.4f} |")

md_lines += [
    "\n### 2.2 Matrice de confusion\n",
    "```",
    f"                  Predit",
    f"                  {class_names[0]:<12}{class_names[1]:<12}",
]
for i, cn in enumerate(class_names):
    row_str = f"Vrai {cn:<10} "
    for j in range(len(class_names)):
        row_str += f"{cm[i][j]:<12}"
    md_lines.append(row_str)
md_lines.append("```\n")

md_lines += [
    f"**Lecture** : sur {len(y_test)} echantillons de test, le modele commet ",
    f"{n_fp + n_fn} erreurs ({n_fp} faux positifs, {n_fn} faux negatifs). ",
    f"Le **recall {class_names[idx_aggr]}** est de **{rec_per[idx_aggr]:.4f}**, ",
    f"ce qui signifie qu'on detecte {100*rec_per[idx_aggr]:.1f}% des vrais cas ",
    f"agressifs - critique pour un systeme d'alerte.\n",
    "## 3. Calibration des seuils d'alerte\n",
    "Le service Flask declenche **deux niveaux d'alerte** bases sur la probabilite ",
    "P(AGGRESSIVE) renvoyee par le modele :\n",
    "| Niveau | Seuil | Action |",
    "|---|---|---|",
    f"| **WARNING** | P >= {SEUIL_WARNING:.2f} | Notification simple |",
    f"| **CRITICAL** | P >= {SEUIL_CRITICAL:.2f} | Alerte au responsable |\n",
    f"### 3.1 Comportement des seuils sur le test\n",
    f"- **{n_warning}** alertes WARNING declenchees sur {len(y_test)} echantillons",
    f"- **{n_critical}** d'entre elles montent au niveau CRITICAL",
    f"- Precision WARNING  : {tp_warning}/{n_warning} = "
    f"**{100*tp_warning/max(n_warning,1):.1f}%** de vraies alertes",
    f"- Precision CRITICAL : {tp_critical}/{n_critical} = "
    f"**{100*tp_critical/max(n_critical,1):.1f}%** de vraies alertes\n",
    "Voir `final_threshold_calibration.png` pour la visualisation.\n",
    "## 4. Analyse des erreurs\n",
    f"Le modele commet **{len(errors_idx)} erreurs** sur {len(y_test)} predictions :",
    f"- **{n_fp} faux positifs** (NORMAL classe AGGRESSIVE) - genere de fausses alertes",
    f"- **{n_fn} faux negatifs** (AGGRESSIVE classe NORMAL) - cas rates (le plus grave)\n",
    f"Voir le detail dans `final_error_analysis.csv`.\n",
    "## 5. Pipeline de production\n",
    "Le service Flask `anomaly_detector.py` charge les artefacts suivants au demarrage :",
    "```",
    "models/",
    "  ├── xgb_driver_behavior.pkl    # Modele XGBoost entraine",
    "  ├── scaler.pkl                  # StandardScaler ajuste sur train",
    "  ├── label_encoder.pkl           # NORMAL <-> 1, AGGRESSIVE <-> 0",
    "  ├── feature_cols.pkl            # Liste des 58 features",
    "  └── alert_thresholds.pkl        # Seuils WARNING / CRITICAL",
    "```\n",
    "**Endpoints REST** :",
    "- `POST /predict` -> recoit les donnees IMU brutes, renvoie {classe, probabilites, niveau_alerte}",
    "- `GET /health`   -> verifie que le service est actif et le modele charge\n",
    "## 6. Verdict production\n",
]
if report["production_ready"]:
    md_lines.append("**[VALIDE] Le modele est pret pour la production** (criteres reunis) :")
    md_lines.append(f"- Gap train-test = {gap:+.4f} < 0.10")
    md_lines.append(f"- Accuracy = {acc:.4f} >= 0.85")
    md_lines.append(f"- AUC = {auc:.4f} >= 0.90")
else:
    md_lines.append("**[A REVOIR] Le modele ne satisfait pas tous les criteres** :")
    md_lines.append(f"- Gap train-test = {gap:+.4f} (cible < 0.10)")
    md_lines.append(f"- Accuracy = {acc:.4f} (cible >= 0.85)")
    md_lines.append(f"- AUC = {auc:.4f} (cible >= 0.90)")

md_lines += [
    "\n## 7. Limites connues et perspectives\n",
    "**Limites actuelles** :",
    "- Test set de petite taille (n=30) -> intervalle de confiance large sur les metriques",
    "- Dataset issu de 2 conducteurs uniquement -> generalisation a d'autres profils a confirmer",
    "- Capteur MPU6050 simule via dataset smartphone -> calibration physique a faire en deploiement\n",
    "**Pistes d'amelioration** :",
    "- Collecter plus de donnees avec le capteur MPU6050 reel monte dans le vehicule",
    "- Ajouter la classe `LOW_NORMAL` (conduite tres lente, possible somnolence)",
    "- Mettre en place un monitoring en production (data drift, performance dans le temps)",
    "- Re-entrainement periodique sur les nouvelles donnees etiquetees\n",
    "## 8. Fichiers livres\n",
    "| Fichier | Description |",
    "|---|---|",
    "| `models/xgb_driver_behavior.pkl` | Modele final entraine |",
    "| `models/scaler.pkl` | Normalisation des features |",
    "| `models/label_encoder.pkl` | Encodage des classes |",
    "| `models/feature_cols.pkl` | Schema des 58 features |",
    "| `models/alert_thresholds.pkl` | Seuils WARNING/CRITICAL |",
    "| `reports/final_evaluation_report.json` | Metriques completes |",
    "| `reports/final_confusion_matrix.png` | Matrice de confusion |",
    "| `reports/final_threshold_calibration.png` | Calibration des seuils |",
    "| `reports/final_error_analysis.csv` | Detail de chaque erreur |",
    "| `reports/model_selection_summary.md` | Justification du choix de XGBoost |",
    "| `oinsexpress_ia/flask_service/anomaly_detector.py` | Service de production |\n",
    "---\n",
    "*Rapport genere automatiquement par `scripts/08_final_evaluation.py`*",
]

with open(RPT_DIR / "final_evaluation_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"      OK final_evaluation_summary.md")

print("\n" + "=" * 70)
print("EVALUATION FINALE TERMINEE")
print("=" * 70)
print(f"Accuracy={acc:.4f} | F1={f1_macro:.4f} | AUC={auc:.4f} | Gap={gap:+.4f}")
verdict = "PRODUCTION-READY" if report["production_ready"] else "A REVOIR"
print(f"Verdict : {verdict}")
print("=" * 70)
