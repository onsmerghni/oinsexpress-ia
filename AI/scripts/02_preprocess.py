"""
PREPROCESSING & FEATURE ENGINEERING — VERSION AVEC SEGMENT SLICING
==================================================================================

CORRECTION MAJEURE :
  Au lieu de 4 gros segments contigus (un par classe), on coupe chaque
  segment en MINI-SEGMENTS de SLICE_SIZE points, puis on melange.
  
  Effets :
    - Casse la "signature" du conducteur/session qui rendait VERY_AGGRESSIVE
      trop facile a reconnaitre (100% suspect)
    - Augmente le nombre de groupes independants pour le CV
    - Reduit l'ecart CV vs Test (ecart structurel)
    - Force le modele a apprendre le COMPORTEMENT et non le SEGMENT

  Resultat attendu : F1-test entre 0.72 et 0.82 + plus de 100% suspect
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not (ROOT / "data").exists():
    import sys
    print(f"ERREUR : dossier 'data' introuvable.", file=sys.stderr)
    sys.exit(1)

RNG = 42
np.random.seed(RNG)

# Parametres
WINDOW = 10
ADD_NOISE = True
NOISE_ACC_STD  = 0.10   # double (avant: 0.05) - simule capteurs degraded
NOISE_GYRO_STD = 0.6    # double (avant: 0.3)  - simule capteurs degraded
TEST_RATIO = 0.20
STRIDE = 5

# ===> NOUVEAU PARAMETRE : taille des mini-segments <===
SLICE_SIZE = 30        # chaque mini-segment fait ~30 points bruts
                       # 1114 points / 30 = ~37 mini-segments au total

# -----------------------------------------------------------------------------
# 1) CHARGEMENT
# -----------------------------------------------------------------------------
print("[1/7] Chargement des donnees...")
df_raw = pd.read_csv(DATA_DIR / "sensor_raw.csv")
df_raw = df_raw.rename(columns={"Target(Class)": "Target"})
print(f"      sensor_raw : {df_raw.shape}")
print(f"      Distribution classes brutes:")
print(df_raw["Target"].value_counts().sort_index().to_string())

# -----------------------------------------------------------------------------
# 2) BRUIT
# -----------------------------------------------------------------------------
if ADD_NOISE:
    print(f"\n[2/7] Ajout de bruit gaussien (acc_std={NOISE_ACC_STD}, gyro_std={NOISE_GYRO_STD})...")
    for c in ["AccX", "AccY", "AccZ"]:
        df_raw[c] = df_raw[c] + np.random.normal(0, NOISE_ACC_STD, len(df_raw))
    for c in ["GyroX", "GyroY", "GyroZ"]:
        df_raw[c] = df_raw[c] + np.random.normal(0, NOISE_GYRO_STD, len(df_raw))
else:
    print("\n[2/7] Bruit desactive.")

# -----------------------------------------------------------------------------
# 3) SEGMENTS PRINCIPAUX (les 4 d'origine)
# -----------------------------------------------------------------------------
print("\n[3/7] Identification des segments contigus principaux...")
df_raw["main_segment_id"] = (df_raw["Target"] != df_raw["Target"].shift()).cumsum()
print(f"      Nombre de segments principaux : {df_raw['main_segment_id'].nunique()}")

# -----------------------------------------------------------------------------
# 3 bis) SEGMENT SLICING ===> NOUVEAU
# -----------------------------------------------------------------------------
print(f"\n[3 bis] Decoupage en mini-segments de {SLICE_SIZE} points...")
df_raw["segment_id"] = -1
slice_counter = 0

for main_sid in df_raw["main_segment_id"].unique():
    mask = df_raw["main_segment_id"] == main_sid
    seg_indices = df_raw[mask].index.tolist()
    n = len(seg_indices)
    # On decoupe ce segment en mini-segments de SLICE_SIZE points chacun
    for start in range(0, n, SLICE_SIZE):
        end = min(start + SLICE_SIZE, n)
        # On evite les mini-segments trop petits (< WINDOW points)
        if end - start < WINDOW:
            # On l'attache au mini-segment precedent
            for idx in seg_indices[start:end]:
                df_raw.at[idx, "segment_id"] = slice_counter - 1
        else:
            for idx in seg_indices[start:end]:
                df_raw.at[idx, "segment_id"] = slice_counter
            slice_counter += 1

n_slices = df_raw["segment_id"].nunique()
print(f"        Nombre de mini-segments crees : {n_slices}")
print(f"        Distribution classe x mini-segments :")
slice_class = df_raw.groupby("segment_id")["Target"].first()
print(slice_class.value_counts().sort_index().to_string())

# -----------------------------------------------------------------------------
# 4) FEATURE ENGINEERING (rolling PAR mini-segment)
# -----------------------------------------------------------------------------
print("\n[4/7] Feature engineering avec rolling PAR mini-segment...")

def add_engineered_features_per_segment(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["AccMag"]  = np.sqrt(df.AccX**2  + df.AccY**2  + df.AccZ**2)
    df["GyroMag"] = np.sqrt(df.GyroX**2 + df.GyroY**2 + df.GyroZ**2)
    df["AccEnergy"]  = df.AccMag ** 2
    df["GyroEnergy"] = df.GyroMag ** 2
    df["GyroAccRatio"] = df.GyroMag / (df.AccMag + 1e-6)

    for c in ["AccX", "AccY", "AccZ"]:
        df[f"Jerk{c[-1]}"] = df.groupby("segment_id")[c].diff().fillna(0)
    df["JerkMag"] = np.sqrt(df.JerkX**2 + df.JerkY**2 + df.JerkZ**2)

    cols_base = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ", "AccMag", "GyroMag"]
    grouped = df.groupby("segment_id")
    for c in cols_base:
        df[f"{c}_roll_mean"]  = grouped[c].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).mean())
        df[f"{c}_roll_std"]   = grouped[c].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).std()).fillna(0)
        df[f"{c}_roll_max"]   = grouped[c].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).max())
        df[f"{c}_roll_min"]   = grouped[c].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).min())
        df[f"{c}_roll_range"] = df[f"{c}_roll_max"] - df[f"{c}_roll_min"]

    df["BrakeIntensity"] = -df.AccX_roll_min
    df["TurnIntensity"]  = df.GyroZ_roll_range
    df["AccelIntensity"] = df.AccX_roll_max

    return df

df_eng = add_engineered_features_per_segment(df_raw)
print(f"      Apres engineering : {df_eng.shape}")

# -----------------------------------------------------------------------------
# 4 bis) STRIDE
# -----------------------------------------------------------------------------
print(f"\n[4 bis] Sous-echantillonnage par stride={STRIDE}...")
sampled_indices = []
for sid, grp in df_eng.groupby("segment_id"):
    seg_idx = grp.index.tolist()
    start = min(WINDOW - 1, len(seg_idx) - 1)
    sampled_indices.extend(seg_idx[start::STRIDE])

df_eng = df_eng.loc[sampled_indices].reset_index(drop=True)
print(f"        Avant : 1114 lignes  ->  Apres : {len(df_eng)} lignes")
print(f"        Distribution classes brutes apres sous-echantillonnage :")
print(df_eng["Target"].value_counts().sort_index().to_string())

# -----------------------------------------------------------------------------
# 4 ter) GROUP_ID = mini-segment (chaque mini-segment = un groupe pour CV)
# -----------------------------------------------------------------------------
print(f"\n[4 ter] Group_id = mini-segment (chaque mini-segment = 1 groupe pour CV)...")
df_eng["group_id"] = df_eng["segment_id"]
print(f"        Nombre de groupes : {df_eng['group_id'].nunique()}")
print(f"        Taille moyenne d'un groupe : {len(df_eng) / df_eng['group_id'].nunique():.1f} lignes")

# -----------------------------------------------------------------------------
# 5) MAPPING BINAIRE + SPLIT PAR MINI-SEGMENTS (au lieu de par segment principal)
# -----------------------------------------------------------------------------
print("\n[5/7] Mapping BINAIRE et split par mini-segments...")

LABEL_MAP = {
    1: "NORMAL",
    2: "NORMAL",
    3: "AGGRESSIVE",
    4: "AGGRESSIVE",
}
df_eng["Behavior"] = df_eng["Target"].map(LABEL_MAP)
df_eng["target_original"] = df_eng["Target"]

print(f"      Distribution business (binaire):")
print(df_eng["Behavior"].value_counts().to_string())

# Split par MINI-SEGMENTS : on prend ~80% des mini-segments en train,
# 20% en test. On garde la stratification par classe pour avoir
# AGGRESSIVE et NORMAL en proportions correctes des deux cotes.
print(f"\n      Split par mini-segments (stratifie par classe brute) :")

unique_slices = df_eng["segment_id"].unique()
slice_to_class = df_eng.groupby("segment_id")["Target"].first().to_dict()

# On melange les slices DETERMINISTIQUEMENT (seed) puis on stratifie par classe
np.random.seed(RNG)
slices_by_class = {c: [] for c in [1, 2, 3, 4]}
for sid in unique_slices:
    slices_by_class[slice_to_class[sid]].append(sid)

train_slices = []
test_slices = []
for cls, slices in slices_by_class.items():
    slices = np.array(slices)
    np.random.shuffle(slices)
    n_test = max(1, int(len(slices) * TEST_RATIO))
    test_slices.extend(slices[:n_test].tolist())
    train_slices.extend(slices[n_test:].tolist())
    print(f"        Classe {cls}: {len(slices)} slices  -> train={len(slices)-n_test}  test={n_test}")

train_slices = set(train_slices)
test_slices = set(test_slices)
print(f"      Total: {len(train_slices)} mini-segments train  /  {len(test_slices)} test")

# Indices = lignes appartenant a chaque ensemble de slices
train_indices = df_eng[df_eng["segment_id"].isin(train_slices)].index.tolist()
test_indices  = df_eng[df_eng["segment_id"].isin(test_slices)].index.tolist()
print(f"      Total lignes: {len(train_indices)} train  /  {len(test_indices)} test")

# Encoder
le = LabelEncoder()
y_all = le.fit_transform(df_eng["Behavior"])
print(f"      Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

META_COLS = ["Target", "Behavior", "segment_id", "main_segment_id", "group_id", "target_original"]
feature_cols = [c for c in df_eng.columns if c not in META_COLS]
X_all = df_eng[feature_cols].values
groups_all = df_eng["group_id"].values
target_orig_all = df_eng["target_original"].values

mask = ~np.isnan(X_all).any(axis=1) & ~np.isinf(X_all).any(axis=1)
valid_train = [i for i in train_indices if mask[i]]
valid_test  = [i for i in test_indices  if mask[i]]

X_train = X_all[valid_train]
y_train = y_all[valid_train]
X_test  = X_all[valid_test]
y_test  = y_all[valid_test]
groups_train = groups_all[valid_train]
target_orig_test = target_orig_all[valid_test]

print(f"\n      X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"      Distribution y_train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"      Distribution y_test : {dict(zip(*np.unique(y_test,  return_counts=True)))}")
print(f"      Distribution classes brutes test:")
unique_orig, counts_orig = np.unique(target_orig_test, return_counts=True)
print(f"        {dict(zip(unique_orig, counts_orig))}")
print(f"      Nombre de groupes en train : {len(np.unique(groups_train))}")

# -----------------------------------------------------------------------------
# 6) NORMALISATION
# -----------------------------------------------------------------------------
print("\n[6/7] Normalisation...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------------------------------------------------------
# 7) SAUVEGARDE
# -----------------------------------------------------------------------------
print("\n[7/7] Sauvegarde...")
np.save(OUT_DIR / "X_train.npy", X_train)
np.save(OUT_DIR / "X_test.npy",  X_test)
np.save(OUT_DIR / "y_train.npy", y_train)
np.save(OUT_DIR / "y_test.npy",  y_test)
np.save(OUT_DIR / "X_train_scaled.npy", X_train_scaled)
np.save(OUT_DIR / "X_test_scaled.npy",  X_test_scaled)
np.save(OUT_DIR / "groups_train.npy", groups_train)
np.save(OUT_DIR / "target_orig_test.npy", target_orig_test)

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(le,     MODEL_DIR / "label_encoder.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

print("      OK : tout sauvegarde.")
print(f"\nClasses finales : {list(le.classes_)}")
print(f"Nombre de features : {len(feature_cols)}")