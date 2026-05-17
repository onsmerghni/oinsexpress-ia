"""
Pipeline complet Driver Behavior AI.

Orchestre l'integralite de la partie IA :
  01 -> exploration des donnees
  02 -> preprocessing + feature engineering
  03 -> entrainement du modele final (XGBoost)
  04 -> inference et systeme d'alertes
  05 -> visualisations
  06 -> benchmark de 7 modeles
  07 -> justification du choix XGBoost (multicriteres)
  08 -> evaluation finale (rapport production-ready)

Usage : python scripts/run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

steps = [
    ("01_explore_data.py",                   "Exploration"),
    ("02_preprocess.py",                     "Preprocessing"),
    ("03_train_xgboost.py",                  "Entrainement XGBoost"),
    ("04_inference_alerts.py",               "Inference + alertes"),
    ("05_visualizations.py",                 "Visualisations"),
    ("06_benchmark_models.py",               "Benchmark 7 modeles"),
    ("07_model_selection_justification.py",  "Justification du choix"),
    ("08_final_evaluation.py",               "Evaluation finale"),
]

print("=" * 70)
print("DRIVER BEHAVIOR AI — Pipeline complet")
print("=" * 70)

for i, (script, desc) in enumerate(steps, 1):
    print(f"\n{'='*70}")
    print(f">>> Etape {i}/{len(steps)} : {desc}  ({script})")
    print(f"{'='*70}")
    r = subprocess.run([sys.executable, str(ROOT / script)])
    if r.returncode != 0:
        print(f"\n!!! Erreur dans {script} (code {r.returncode}). Arret du pipeline.")
        sys.exit(1)

print("\n" + "=" * 70)
print("[OK] Pipeline complet termine avec succes.")
print("=" * 70)
print("\nFichiers cles produits :")
print("  - models/xgb_driver_behavior.pkl        (modele de production)")
print("  - reports/training_report.json          (metriques d'entrainement)")
print("  - reports/benchmark_results.csv         (comparaison 7 modeles)")
print("  - reports/model_selection_summary.md    (justification de XGBoost)")
print("  - reports/final_evaluation_summary.md   (cloture de la partie IA)")
print("  - reports/final_evaluation_report.json  (metriques officielles)")
