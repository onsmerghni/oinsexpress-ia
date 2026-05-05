"""
OINSExpress — Microservice Flask XGBoost (modèle réel)
=======================================================
Modèle entraîné sur données réelles (driver behavior dataset).
Classification binaire : NORMAL | AGGRESSIVE (avec 3 niveaux de sévérité)

Endpoints :
  GET  /health   → statut du service
  POST /predict  → classification IMU temps réel

Format requête POST /predict :
{
  "livreurId": "LIV-001",
  "raw_imu": {
    "accX": 0.05, "accY": 0.12, "accZ": -0.98,
    "gyrX": 2.3,  "gyrY": 1.1,  "gyrZ": 8.5
  }
}

Format réponse :
{
  "drivingState": "NORMAL",      // NORMAL | AGGRESSIVE
  "severity":     "NONE",        // NONE | MEDIUM | HIGH
  "message":      "Conduite normale",
  "proba":        0.08,          // P(AGGRESSIVE)
  "bufferSize":   5              // lectures accumulées dans la fenêtre
}
"""

import os
import json
import logging
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify
from features import FeatureExtractor
from pathlib import Path

# ── Logging ──
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app     = Flask(__name__)
MODEL_DIR = Path(__file__).parent / "models"

# ══════════════════════════════════════════════════════
# CHARGEMENT DES ARTEFACTS
# ══════════════════════════════════════════════════════

def load_artifacts():
    # ── Modèle XGBoost (format natif JSON — pas besoin de sklearn) ──
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "xgb_driver_behavior.json"))
    log.info("✅ Modèle XGBoost chargé (%s)", MODEL_DIR / "xgb_driver_behavior.json")

    # ── Feature columns (ordre exact d'entraînement) ──
    feature_cols = json.loads((MODEL_DIR / "feature_cols.json").read_text())
    log.info("✅ %d features chargées", len(feature_cols))

    # ── Scaler (StandardScaler sans sklearn) ──
    sp = json.loads((MODEL_DIR / "scaler_params.json").read_text())
    scaler_mean  = np.array(sp["mean_"],  dtype=np.float64)
    scaler_scale = np.array(sp["scale_"], dtype=np.float64)
    log.info("✅ Scaler chargé")

    # ── Seuils d'alerte ──
    th = json.loads((MODEL_DIR / "alert_thresholds.json").read_text())
    seuil_warning  = float(th.get("warning",  0.50))
    seuil_critical = float(th.get("critical", 0.90))
    log.info("✅ Seuils : WARNING=%.2f  CRITICAL=%.2f",
             seuil_warning, seuil_critical)

    # ── Label encoder ──
    le = json.loads((MODEL_DIR / "label_encoder.json").read_text())
    classes     = le["classes_"]          # ['AGGRESSIVE', 'NORMAL']
    idx_aggr    = classes.index("AGGRESSIVE")

    return booster, feature_cols, scaler_mean, scaler_scale, \
           seuil_warning, seuil_critical, idx_aggr


booster, FEATURE_COLS, SCALER_MEAN, SCALER_SCALE, \
    SEUIL_WARNING, SEUIL_CRITICAL, IDX_AGGR = load_artifacts()

extractor = FeatureExtractor(window=10)


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def features_to_vector(feat_dict: dict) -> np.ndarray:
    """Convertit le dict de features en vecteur numpy dans l'ordre du modèle."""
    return np.array([[feat_dict.get(c, 0.0) for c in FEATURE_COLS]],
                    dtype=np.float64)


def scale(x: np.ndarray) -> np.ndarray:
    """Applique la normalisation StandardScaler sans sklearn."""
    return (x - SCALER_MEAN) / (SCALER_SCALE + 1e-12)


def classify(p_aggr: float) -> tuple[str, str, str]:
    """Retourne (drivingState, severity, message) selon les seuils calibrés."""
    if p_aggr < SEUIL_WARNING:
        return "NORMAL", "NONE", "Conduite normale"
    elif p_aggr < SEUIL_CRITICAL:
        return "AGGRESSIVE", "MEDIUM", \
               "Conduite agressive détectée — surveillance recommandée"
    else:
        return "AGGRESSIVE", "HIGH", \
               "Conduite TRÈS agressive — intervention urgente"


# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model':        'XGBoost-OINSExpress-RealData',
        'features':     len(FEATURE_COLS),
        'thresholds':   {'warning': SEUIL_WARNING, 'critical': SEUIL_CRITICAL},
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Corps JSON requis'}), 400

    livreur_id = data.get('livreurId', 'unknown')
    raw_imu    = data.get('raw_imu', {})

    if not raw_imu:
        return jsonify({'error': 'raw_imu manquant'}), 400

    # ── 1. Ajouter la lecture au buffer ──
    extractor.add_reading(livreur_id, raw_imu)
    buf_size = extractor.buffer_size(livreur_id)

    # ── 2. Extraire les 58 features ──
    feat_dict = extractor.get_features(livreur_id)
    if not feat_dict:
        return jsonify({'error': 'Buffer vide'}), 400

    # ── 3. Mise en forme + normalisation ──
    x_raw    = features_to_vector(feat_dict)
    x_scaled = scale(x_raw)

    # ── 4. Prédiction ──
    dmatrix   = xgb.DMatrix(x_scaled)
    proba_raw = booster.predict(dmatrix)[0]   # shape scalaire ou (2,) selon config

    # Gérer les deux cas : softprob (tableau) ou sigmoid (scalaire)
    if hasattr(proba_raw, '__len__'):
        p_aggr = float(proba_raw[IDX_AGGR])
    else:
        # Booster binaire → sortie sigmoid → probabilité de la classe 1
        # Selon l'ordre du LabelEncoder : 0=AGGRESSIVE, 1=NORMAL
        # XGBoost binaire prédit P(classe positive = 1 = NORMAL)
        # → P(AGGRESSIVE) = 1 - sortie
        p_aggr = float(1.0 - proba_raw)

    # ── 5. Seuillage → état ──
    state, severity, message = classify(p_aggr)

    log.info("[%s] buf=%d  P(AGGR)=%.3f  → %s (%s)",
             livreur_id, buf_size, p_aggr, state, severity)

    return jsonify({
        'drivingState': state,
        'severity':     severity,
        'message':      message,
        'proba':        round(p_aggr, 4),
        'bufferSize':   buf_size,
    })


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
