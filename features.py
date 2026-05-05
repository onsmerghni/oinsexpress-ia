"""
OINSExpress — Feature Engineering pour XGBoost
================================================
Maintient un buffer glissant de 5 lectures IMU par livreur.
Calcule 34 features statistiques pour la classification.
"""

import numpy as np
from collections import defaultdict, deque

WINDOW_SIZE = 5  # 5 lectures × 2s = 10 secondes de contexte
SENSORS = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']

# Limites physiques pour filtrer les valeurs aberrantes
CLIP_ACC = 5.0    # ±5g max
CLIP_GYR = 500.0  # ±500°/s max


class FeatureExtractor:
    """Extrait les features XGBoost depuis un buffer glissant par livreur."""

    def __init__(self):
        self.buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def add_reading(self, livreur_id: str, raw_imu: dict):
        """Ajoute une lecture IMU au buffer du livreur."""
        reading = [
            float(np.clip(raw_imu.get('accX', 0.0), -CLIP_ACC, CLIP_ACC)),
            float(np.clip(raw_imu.get('accY', 0.0), -CLIP_ACC, CLIP_ACC)),
            float(np.clip(raw_imu.get('accZ', 1.0), -CLIP_ACC, CLIP_ACC)),
            float(np.clip(raw_imu.get('gyrX', 0.0), -CLIP_GYR, CLIP_GYR)),
            float(np.clip(raw_imu.get('gyrY', 0.0), -CLIP_GYR, CLIP_GYR)),
            float(np.clip(raw_imu.get('gyrZ', 0.0), -CLIP_GYR, CLIP_GYR)),
        ]
        self.buffers[livreur_id].append(reading)

    def get_features(self, livreur_id: str) -> list:
        """
        Retourne le vecteur de 34 features pour le livreur.
        Si le buffer est vide → vecteur nul (NORMAL par défaut).
        """
        buf = list(self.buffers[livreur_id])
        if not buf:
            return [0.0] * 34

        # Remplir le buffer avec la dernière valeur si < WINDOW_SIZE
        while len(buf) < WINDOW_SIZE:
            buf.insert(0, buf[0])

        arr = np.array(buf, dtype=np.float32)  # shape: (5, 6)

        features = []

        # ── 24 features statistiques (4 stats × 6 capteurs) ──
        for i in range(6):
            col = arr[:, i]
            abs_col = np.abs(col)
            features.append(float(np.mean(col)))         # moyenne
            features.append(float(np.std(col)))          # écart-type
            features.append(float(np.max(abs_col)))      # max absolu (pic)
            features.append(float(np.mean(abs_col)))     # moyenne absolue

        # ── 10 features dérivées ──
        acc = arr[:, :3]   # accX, accY, accZ
        gyr = arr[:, 3:]   # gyrX, gyrY, gyrZ

        acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))  # norme accélération
        gyr_mag = np.sqrt(np.sum(gyr ** 2, axis=1))  # norme gyroscope

        features.append(float(np.mean(acc_mag)))          # norme acc moyenne
        features.append(float(np.max(acc_mag)))           # norme acc pic
        features.append(float(np.mean(gyr_mag)))          # norme gyr moyenne
        features.append(float(np.max(gyr_mag)))           # norme gyr pic
        features.append(float(np.mean(np.abs(arr[:, 1])))) # |accY| moy (latéral)
        features.append(float(np.max(np.abs(arr[:, 1]))))  # |accY| max (latéral)
        features.append(float(np.mean(np.abs(arr[:, 5])))) # |gyrZ| moy (lacet)
        features.append(float(np.max(np.abs(arr[:, 5]))))  # |gyrZ| max (lacet)
        features.append(float(np.std(acc_mag)))            # variabilité acc
        features.append(float(np.std(gyr_mag)))            # variabilité gyr

        return features  # 34 features au total


def get_feature_names() -> list:
    """Retourne les noms des 34 features (pour logs/debug)."""
    names = []
    stats = ['mean', 'std', 'max_abs', 'mean_abs']
    for sensor in SENSORS:
        for stat in stats:
            names.append(f"{sensor}_{stat}")
    names += [
        'acc_mag_mean', 'acc_mag_max',
        'gyr_mag_mean', 'gyr_mag_max',
        'accY_abs_mean', 'accY_abs_max',
        'gyrZ_abs_mean', 'gyrZ_abs_max',
        'acc_mag_std', 'gyr_mag_std',
    ]
    return names
