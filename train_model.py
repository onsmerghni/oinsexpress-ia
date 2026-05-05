"""
train_model.py — OINSExpress
============================
Le modèle est désormais PRÉ-ENTRAÎNÉ sur des données réelles
(driver behavior dataset, 1114 échantillons, F1=0.90 sur le test set).

Ce script vérifie uniquement que les artefacts sont présents.
"""

from pathlib import Path
import sys

MODEL_DIR = Path(__file__).parent / "models"
REQUIRED = [
    "xgb_driver_behavior.json",
    "feature_cols.json",
    "scaler_params.json",
    "alert_thresholds.json",
    "label_encoder.json",
]

print("═" * 50)
print("  OINSExpress — Vérification des artefacts ML")
print("═" * 50)

missing = [f for f in REQUIRED if not (MODEL_DIR / f).exists()]

if missing:
    print(f"[ERREUR] Fichiers manquants dans models/ :")
    for f in missing:
        print(f"  - {f}")
    print("\n  → Lance convert_models.py localement puis git push")
    sys.exit(1)

print("[OK] Tous les artefacts sont présents :")
for f in REQUIRED:
    size = (MODEL_DIR / f).stat().st_size
    print(f"  ✓  {f}  ({size:,} bytes)")
print("═" * 50)
