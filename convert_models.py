# convert_models.py -- A lancer UNE FOIS localement
# Convertit les fichiers .pkl scikit-learn en JSON
# pour que le service Flask n'ait plus besoin de scikit-learn.
# Usage : python convert_models.py

import joblib
import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

print("═" * 50)
print("  Conversion des artefacts ML → JSON")
print("═" * 50)

# ── 1. Scaler ──
sc = joblib.load(MODEL_DIR / "scaler.pkl")
out = {"mean_": sc.mean_.tolist(), "scale_": sc.scale_.tolist()}
(MODEL_DIR / "scaler_params.json").write_text(json.dumps(out, indent=2))
print(f"[OK] scaler_params.json   ({len(sc.mean_)} features)")

# ── 2. Feature columns ──
fc = joblib.load(MODEL_DIR / "feature_cols.pkl")
(MODEL_DIR / "feature_cols.json").write_text(json.dumps(fc, indent=2))
print(f"[OK] feature_cols.json    ({len(fc)} features)")
print(f"     Premières : {fc[:4]} ...")
print(f"     Dernières : ... {fc[-3:]}")

# ── 3. Alert thresholds ──
th = joblib.load(MODEL_DIR / "alert_thresholds.pkl")
(MODEL_DIR / "alert_thresholds.json").write_text(json.dumps(th, indent=2))
print(f"[OK] alert_thresholds.json  {th}")

# ── 4. Label encoder ──
le = joblib.load(MODEL_DIR / "label_encoder.pkl")
le_out = {"classes_": le.classes_.tolist()}
(MODEL_DIR / "label_encoder.json").write_text(json.dumps(le_out, indent=2))
print(f"[OK] label_encoder.json   classes={le.classes_.tolist()}")

print("═" * 50)
print("[DONE] Tous les fichiers JSON créés dans models/")
print("       Tu peux maintenant faire : git add models/*.json && git push")
