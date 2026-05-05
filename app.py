"""
OINSExpress — Microservice Flask XGBoost
=========================================
Classification temps réel du comportement de conduite.

Endpoints :
  GET  /health   → statut du service
  POST /predict  → classification IMU

Format requête POST /predict :
{
  "livreurId": "LIV-001",
  "raw_imu": {
    "accX": 0.05, "accY": 0.12, "accZ": 1.01,
    "gyrX": 2.3,  "gyrY": 1.1,  "gyrZ": 8.5
  }
}

Format réponse :
{
  "drivingState": "NORMAL",   // NORMAL | RISKY | AGGRESSIVE
  "severity":     "NONE",     // NONE   | MEDIUM | HIGH
  "message":      "Conduite normale",
  "proba":        0.92
}
"""

import os
import pickle
import logging
from flask import Flask, request, jsonify
from features import FeatureExtractor

# ── Logging ──
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Chargement du modèle ──
MODEL_PATH = 'model.pkl'

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        log.info("✅ Modèle XGBoost chargé depuis %s", MODEL_PATH)
        return model
    else:
        log.warning("⚠️  model.pkl introuvable — entraînement en cours...")
        from train_model import train_and_save
        return train_and_save(MODEL_PATH)

model    = load_model()
extractor = FeatureExtractor()

STATES = ['NORMAL', 'RISKY', 'AGGRESSIVE']
SEVERITY = {'NORMAL': 'NONE', 'RISKY': 'MEDIUM', 'AGGRESSIVE': 'HIGH'}
MESSAGES = {
    'NORMAL':     'Conduite normale',
    'RISKY':      'Conduite risquée détectée — vigilance recommandée',
    'AGGRESSIVE': 'Conduite agressive détectée — intervention requise',
}


# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model':  'XGBoost-OINSExpress',
        'classes': STATES,
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

    # Ajouter la lecture au buffer du livreur
    extractor.add_reading(livreur_id, raw_imu)

    # Extraire les features
    features = extractor.get_features(livreur_id)

    # Prédiction XGBoost
    proba_arr = model.predict_proba([features])[0]
    pred_idx  = int(proba_arr.argmax())
    state     = STATES[pred_idx]
    proba     = float(proba_arr[pred_idx])

    log.info("[%s] → %s (proba=%.2f)", livreur_id, state, proba)

    return jsonify({
        'drivingState': state,
        'severity':     SEVERITY[state],
        'message':      MESSAGES[state],
        'proba':        round(proba, 3),
    })


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
