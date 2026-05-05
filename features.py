"""
features.py — OINSExpress Feature Extractor (temps réel)
=========================================================
Reproduit EXACTEMENT le pipeline de 02_preprocess.py :
  - window=10 rolling stats pour 8 signaux de base
  - Features dérivées : AccMag, GyroMag, Jerk*, énergie, ratio
  - BrakeIntensity, TurnIntensity, AccelIntensity

Résultat : 58 features dans le même ordre que feature_cols.json

PFA 2026 — Mrabet Islem & Merghni Ons
"""

from collections import deque
import numpy as np

WINDOW = 10   # doit correspondre au paramètre d'entraînement


class FeatureExtractor:
    """
    Maintient un buffer glissant par livreur.
    Chaque appel à add_reading() + get_features() produit
    le vecteur de 58 features prêt pour le modèle.
    """

    def __init__(self, window: int = WINDOW):
        self.window = window
        self._buffers: dict = {}

    # ──────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────

    def add_reading(self, livreur_id: str, reading: dict) -> None:
        """
        Ajoute une lecture IMU au buffer du livreur.
        Accepte les deux conventions de nommage :
          - accX / accY / accZ / gyrX / gyrY / gyrZ  (ESP32 / Spring Boot)
          - AccX / AccY / AccZ / GyroX / GyroY / GyroZ (dataset original)
        """
        if livreur_id not in self._buffers:
            self._buffers[livreur_id] = deque(maxlen=self.window)

        r = {
            'AccX':  float(reading.get('accX',  reading.get('AccX',  0.0))),
            'AccY':  float(reading.get('accY',  reading.get('AccY',  0.0))),
            'AccZ':  float(reading.get('accZ',  reading.get('AccZ',  0.0))),
            'GyroX': float(reading.get('gyrX',  reading.get('GyroX', 0.0))),
            'GyroY': float(reading.get('gyrY',  reading.get('GyroY', 0.0))),
            'GyroZ': float(reading.get('gyrZ',  reading.get('GyroZ', 0.0))),
        }
        self._buffers[livreur_id].append(r)

    def get_features(self, livreur_id: str) -> dict:
        """
        Retourne un dict {feature_name: value} avec 58 features.
        Retourne {} si le buffer est vide.
        """
        buf = list(self._buffers.get(livreur_id, []))
        if not buf:
            return {}
        return self._compute(buf)

    def buffer_size(self, livreur_id: str) -> int:
        return len(self._buffers.get(livreur_id, []))

    # ──────────────────────────────────────────────────────────
    # Calcul des features (miroir exact de 02_preprocess.py)
    # ──────────────────────────────────────────────────────────

    def _compute(self, buf: list) -> dict:
        n = len(buf)

        accX  = np.array([r['AccX']  for r in buf])
        accY  = np.array([r['AccY']  for r in buf])
        accZ  = np.array([r['AccZ']  for r in buf])
        gyrX  = np.array([r['GyroX'] for r in buf])
        gyrY  = np.array([r['GyroY'] for r in buf])
        gyrZ  = np.array([r['GyroZ'] for r in buf])

        # ── Magnitudes & énergie ──
        accMag       = np.sqrt(accX**2 + accY**2 + accZ**2)
        gyroMag      = np.sqrt(gyrX**2 + gyrY**2 + gyrZ**2)
        accEnergy    = accMag  ** 2
        gyroEnergy   = gyroMag ** 2
        gyroAccRatio = gyroMag / (accMag + 1e-6)

        # ── Jerk (différence finie sur le dernier point) ──
        jerkX   = float(accX[-1] - accX[-2]) if n >= 2 else 0.0
        jerkY   = float(accY[-1] - accY[-2]) if n >= 2 else 0.0
        jerkZ   = float(accZ[-1] - accZ[-2]) if n >= 2 else 0.0
        jerkMag = float(np.sqrt(jerkX**2 + jerkY**2 + jerkZ**2))

        feat = {
            # Valeurs instantanées (dernier point)
            'GyroX':        float(gyrX[-1]),
            'GyroY':        float(gyrY[-1]),
            'GyroZ':        float(gyrZ[-1]),
            'AccX':         float(accX[-1]),
            'AccY':         float(accY[-1]),
            'AccZ':         float(accZ[-1]),
            # Dérivées scalaires
            'AccMag':       float(accMag[-1]),
            'GyroMag':      float(gyroMag[-1]),
            'AccEnergy':    float(accEnergy[-1]),
            'GyroEnergy':   float(gyroEnergy[-1]),
            'GyroAccRatio': float(gyroAccRatio[-1]),
            # Jerk
            'JerkX':   jerkX,
            'JerkY':   jerkY,
            'JerkZ':   jerkZ,
            'JerkMag': jerkMag,
        }

        # ── Rolling stats (même ordre que cols_base dans 02_preprocess.py) ──
        signals = {
            'AccX':    accX,
            'AccY':    accY,
            'AccZ':    accZ,
            'GyroX':   gyrX,
            'GyroY':   gyrY,
            'GyroZ':   gyrZ,
            'AccMag':  accMag,
            'GyroMag': gyroMag,
        }
        for cname, arr in signals.items():
            # pandas rolling().std() utilise ddof=1 par défaut
            std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            feat[f'{cname}_roll_mean']  = float(np.mean(arr))
            feat[f'{cname}_roll_std']   = std_val
            feat[f'{cname}_roll_max']   = float(np.max(arr))
            feat[f'{cname}_roll_min']   = float(np.min(arr))
            feat[f'{cname}_roll_range'] = float(np.max(arr) - np.min(arr))

        # ── Features dérivées du rolling ──
        feat['BrakeIntensity'] = -feat['AccX_roll_min']
        feat['TurnIntensity']  =  feat['GyroZ_roll_range']
        feat['AccelIntensity'] =  feat['AccX_roll_max']

        return feat
