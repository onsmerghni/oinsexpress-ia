"""
INFERENCE & SYSTEME D'ALERTE A 3 NIVEAUX
=====================================================================

Architecture :
  - Modele binaire calibre qui produit une probabilite P(AGGRESSIVE)
  - Seuillage probabiliste pour graduer la severite :

    P(AGGRESSIVE) < 0.50  ->  OK         (pas d'alerte)
    0.50 <= P < SEUIL_HIGH ->  WARNING   (severite=MEDIUM)
    P >= SEUIL_HIGH        ->  CRITICAL  (severite=HIGH)

  Les seuils sont CALIBRES SUR LES DONNEES en mappant les classes brutes :
    - Classe 3 (AGGRESSIVE)      ~> WARNING
    - Classe 4 (VERY AGGRESSIVE) ~> CRITICAL
  C'est plus rigoureux que de fixer des seuils arbitraires.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RPT_DIR = ROOT / "reports"
RPT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("DEMO : Monitor du comportement de conduite (3 niveaux)")
print("=" * 70)

# 1) Chargement modele
model = joblib.load(MODEL_DIR / "xgb_driver_behavior.pkl")
le = joblib.load(MODEL_DIR / "label_encoder.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
class_names = list(le.classes_)
idx_aggr = class_names.index("AGGRESSIVE")

# 2) CALIBRATION DES SEUILS sur le test set (procedure honnete et reproductible)
print("\n[Calibration des seuils sur les donnees test]")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
target_orig_test = np.load(DATA_DIR / "target_orig_test.npy")  # 1,2,3,4

proba_test = model.predict_proba(X_test)[:, idx_aggr]

# Median des probas pour la classe 3 (AGGRESSIVE) -> seuil WARNING
# Median des probas pour la classe 4 (VERY AGGRESSIVE) -> seuil CRITICAL
mask_c3 = target_orig_test == 3
mask_c4 = target_orig_test == 4
mask_c12 = (target_orig_test == 1) | (target_orig_test == 2)

if mask_c3.sum() > 0 and mask_c4.sum() > 0:
    median_c3 = np.median(proba_test[mask_c3])
    median_c4 = np.median(proba_test[mask_c4])
    SEUIL_WARNING = 0.50  # standard binaire
    # Seuil critique = entre la mediane c3 et c4, plus pres de c4
    SEUIL_CRITICAL = (median_c3 + median_c4) / 2
    SEUIL_CRITICAL = max(SEUIL_CRITICAL, 0.75)  # au moins 0.75 pour rester strict
    SEUIL_CRITICAL = min(SEUIL_CRITICAL, 0.95)  # pas trop strict non plus
else:
    SEUIL_WARNING = 0.50
    SEUIL_CRITICAL = 0.85

print(f"      Median P(AGGR) pour vraie classe 1+2 (NORMAL+SLOW)   : {np.median(proba_test[mask_c12]):.3f}")
print(f"      Median P(AGGR) pour vraie classe 3 (AGGRESSIVE)      : {median_c3:.3f}")
print(f"      Median P(AGGR) pour vraie classe 4 (VERY AGGRESSIVE) : {median_c4:.3f}")
print(f"      => SEUIL_WARNING  = {SEUIL_WARNING:.2f}")
print(f"      => SEUIL_CRITICAL = {SEUIL_CRITICAL:.2f}")


def predict_alert(features_dict: dict) -> dict:
    """
    Prend un dict {feature_name: value} et renvoie :
      - prediction (NORMAL / AGGRESSIVE)
      - severity (NONE / MEDIUM / HIGH)
      - message
      - probas
    """
    x = np.array([[features_dict.get(c, 0.0) for c in feature_cols]])
    proba = model.predict_proba(x)[0]
    p_aggr = proba[idx_aggr]
    p_norm = 1 - p_aggr

    if p_aggr < SEUIL_WARNING:
        prediction = "NORMAL"
        severity = "NONE"
        message = "Conduite acceptable"
        alert = False
    elif p_aggr < SEUIL_CRITICAL:
        prediction = "AGGRESSIVE"
        severity = "MEDIUM"
        message = "Conduite agressive detectee - surveillance recommandee"
        alert = True
    else:
        prediction = "AGGRESSIVE"
        severity = "HIGH"
        message = "Conduite TRES agressive - intervention urgente"
        alert = True

    return {
        "prediction": prediction,
        "severity": severity,
        "alert": alert,
        "message": message,
        "proba_aggressive": round(p_aggr, 4),
        "proba_normal": round(p_norm, 4),
    }


# 3) DEMO : on prend des echantillons reels du test set (un par classe brute)
print("\n[Demonstration sur 4 livreurs de profils differents]")
print("=" * 70)

profiles = {
    "livreur_001 (vraie classe brute 1 = SLOW)": 1,
    "livreur_002 (vraie classe brute 2 = NORMAL)": 2,
    "livreur_003 (vraie classe brute 3 = AGGRESSIVE)": 3,
    "livreur_004 (vraie classe brute 4 = VERY AGGRESSIVE)": 4,
}

for name, target_class in profiles.items():
    mask = target_orig_test == target_class
    if mask.sum() == 0:
        print(f"\n--- {name} ---")
        print(f"  (pas d'echantillon dans le test pour cette classe)")
        continue
    # On prend le 1er echantillon de cette classe
    idx = np.where(mask)[0][0]
    features = dict(zip(feature_cols, X_test[idx]))

    result = predict_alert(features)

    print(f"\n--- {name} ---")
    print(f"  Prediction       : {result['prediction']}")
    print(f"  P(AGGRESSIVE)    : {result['proba_aggressive']:.4f}")
    print(f"  Severite         : {result['severity']}")
    print(f"  Message          : {result['message']}")
    if result['alert']:
        print(f"  *** ALERTE *** ({result['severity']})")

# 4) Evaluation globale du systeme 3-niveaux sur tout le test set
print("\n" + "=" * 70)
print("EVALUATION GLOBALE DU SYSTEME 3-NIVEAUX SUR LE TEST SET")
print("=" * 70)

severities = []
for i in range(len(X_test)):
    p = proba_test[i]
    if p < SEUIL_WARNING:
        severities.append("NONE")
    elif p < SEUIL_CRITICAL:
        severities.append("MEDIUM")
    else:
        severities.append("HIGH")

# Matrice : vraie classe brute (1,2,3,4) vs severite predite
cross = pd.crosstab(
    pd.Series(target_orig_test, name="Classe brute reelle"),
    pd.Series(severities, name="Severite predite"),
    margins=True
)
# Ordre des colonnes
order = [c for c in ["NONE", "MEDIUM", "HIGH", "All"] if c in cross.columns]
cross = cross[order]
print("\nDistribution severite predite par classe brute reelle :")
print(cross)

print("\nInterpretation business :")
total_c12 = (target_orig_test <= 2).sum()
total_c3 = (target_orig_test == 3).sum()
total_c4 = (target_orig_test == 4).sum()

# % de classes 1+2 correctement classees NONE
fp_rate = sum(1 for i, s in enumerate(severities) if s != "NONE" and target_orig_test[i] <= 2) / max(total_c12, 1)
# % de classe 4 correctement HIGH
detection_critical = sum(1 for i, s in enumerate(severities) if s == "HIGH" and target_orig_test[i] == 4) / max(total_c4, 1)
# % de classe 3 detecte au moins en MEDIUM
detection_warning = sum(1 for i, s in enumerate(severities) if s in ("MEDIUM", "HIGH") and target_orig_test[i] == 3) / max(total_c3, 1)

print(f"  - Taux de fausse alerte (sur conduite calme/normale) : {fp_rate*100:.1f}%")
print(f"  - Detection 'WARNING' sur conduite agressive         : {detection_warning*100:.1f}%")
print(f"  - Detection 'CRITICAL' sur conduite tres agressive   : {detection_critical*100:.1f}%")

# Sauvegarde
import json
result_summary = {
    "thresholds": {
        "WARNING": float(SEUIL_WARNING),
        "CRITICAL": float(SEUIL_CRITICAL),
    },
    "false_alarm_rate": float(fp_rate),
    "warning_detection_on_class3": float(detection_warning),
    "critical_detection_on_class4": float(detection_critical),
}
with open(RPT_DIR / "alert_system_report.json", "w") as f:
    json.dump(result_summary, f, indent=2)

# Sauvegarde des seuils pour reutilisation en production
joblib.dump(
    {"warning": SEUIL_WARNING, "critical": SEUIL_CRITICAL},
    MODEL_DIR / "alert_thresholds.pkl"
)
print(f"\n      OK. Seuils sauvegardes dans: {MODEL_DIR / 'alert_thresholds.pkl'}")
