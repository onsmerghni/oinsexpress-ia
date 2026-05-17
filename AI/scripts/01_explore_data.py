"""
Exploration des données brutes du dataset Driver Behavior.

Dataset source: jair-jr/driverBehaviorDataset (capteurs smartphone réels)
Distribué via Kaggle, données issues d'expériences réelles avec vidéos
servant à étiqueter les comportements de conduite.

Classes:
    1 = SLOW (conduite calme / lente)
    2 = NORMAL (conduite normale)
    3 = AGGRESSIVE (conduite agressive)
    4 = VERY_AGGRESSIVE / DANGEROUS

Capteurs:
    GyroX/Y/Z = vitesse angulaire (rotation)
    AccX/Y/Z  = accélération linéaire (g)
"""

import pandas as pd
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT / "data")

# Vérification des chemins (diagnostic clair en cas de problème)
if not (ROOT / "data").exists():
    import sys
    print(f"ERREUR : dossier 'data' introuvable.", file=sys.stderr)
    print(f"Chemin attendu : {ROOT / 'data'}", file=sys.stderr)
    print(f"Script lance depuis : {Path.cwd()}", file=sys.stderr)
    print(f"Verifiez que vous avez extrait le dossier driver_ai au complet", file=sys.stderr)
    print(f"et que vous lancez ce script depuis le bon emplacement.", file=sys.stderr)
    sys.exit(1)

print("=" * 70)
print("EXPLORATION DU DATASET DRIVER BEHAVIOR")
print("=" * 70)

# 1. Données brutes (capteurs)
df_raw = pd.read_csv(f"{DATA_DIR}/sensor_raw.csv")
print(f"\n[1] sensor_raw.csv — données brutes")
print(f"    Shape: {df_raw.shape}")
print(f"    Colonnes: {list(df_raw.columns)}")
print(f"\n    Distribution des classes:")
print(df_raw["Target(Class)"].value_counts().sort_index())
print(f"\n    Statistiques descriptives:")
print(df_raw.describe().round(3))
print(f"\n    Valeurs manquantes: {df_raw.isnull().sum().sum()}")
print(f"    Doublons: {df_raw.duplicated().sum()}")

# 2. Features pré-calculées
df_feat = pd.read_csv(f"{DATA_DIR}/features_14.csv")
print(f"\n[2] features_14.csv — features statistiques pré-calculées")
print(f"    Shape: {df_feat.shape}")
print(f"    Nombre de features: {df_feat.shape[1] - 1}")
print(f"\n    Distribution des classes:")
print(df_feat["Target"].value_counts().sort_index())
print(f"\n    Valeurs manquantes: {df_feat.isnull().sum().sum()}")

# 3. Analyse des plages par classe (data brutes)
print(f"\n[3] Analyse comportementale par classe (donnees brutes)")
print(f"    Classe 1 = SLOW, 2 = NORMAL, 3 = AGGRESSIVE, 4 = VERY AGGRESSIVE\n")

for c in sorted(df_raw["Target(Class)"].unique()):
    sub = df_raw[df_raw["Target(Class)"] == c]
    print(f"  Classe {c} ({len(sub)} echantillons):")
    print(f"    AccX  range=[{sub['AccX'].min():+.3f}, {sub['AccX'].max():+.3f}]  std={sub['AccX'].std():.3f}")
    print(f"    AccY  range=[{sub['AccY'].min():+.3f}, {sub['AccY'].max():+.3f}]  std={sub['AccY'].std():.3f}")
    print(f"    AccZ  range=[{sub['AccZ'].min():+.3f}, {sub['AccZ'].max():+.3f}]  std={sub['AccZ'].std():.3f}")
    print(f"    GyroX std={sub['GyroX'].std():.3f}  GyroY std={sub['GyroY'].std():.3f}  GyroZ std={sub['GyroZ'].std():.3f}")
    print()

print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)
print("""
Le dataset est REEL (capteurs smartphone, etiquetage video manuel).
Il convient pour entrainer un classifieur de comportement de conduite.

Limitations observees :
  - Taille modeste (~1100 echantillons)
  - 4 classes deja deployes mais 'features_14' est juste 'sensor_raw' agrege
  - On va donc reconstruire des features riches en fenetres glissantes
    sur les donnees brutes pour avoir un signal plus exploitable
""")
