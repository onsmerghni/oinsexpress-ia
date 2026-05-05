"""
OINSExpress — Entraînement du modèle XGBoost
=============================================
Génère des données synthétiques réalistes basées sur la physique
de la conduite et entraîne un XGBClassifier 3 classes :
  0 = NORMAL | 1 = RISKY | 2 = AGGRESSIVE

Lancé automatiquement lors du build Render.
"""

import numpy as np
import pickle
import xgboost as xgb
from features import FeatureExtractor

np.random.seed(42)


def simulate_window(acc_scale: float, gyr_scale: float, n: int) -> np.ndarray:
    """
    Génère n vecteurs de features simulant une fenêtre de conduite.

    acc_scale : intensité de l'accélération (1=normal, 3=risqué, 6=agressif)
    gyr_scale : intensité du gyroscope     (1=normal, 4=risqué, 10=agressif)
    """
    extractor = FeatureExtractor()
    result = []

    for _ in range(n):
        # Simuler WINDOW_SIZE=5 lectures pour ce style de conduite
        livreur_id = f"sim_{np.random.randint(1000)}"
        for _ in range(5):
            reading = {
                'accX': float(np.random.normal(0, 0.05 * acc_scale)),
                'accY': float(np.random.normal(0, 0.18 * acc_scale)),
                'accZ': float(np.random.normal(1.0, 0.04)),
                'gyrX': float(np.random.normal(0, 8.0  * gyr_scale)),
                'gyrY': float(np.random.normal(0, 6.0  * gyr_scale)),
                'gyrZ': float(np.random.normal(0, 12.0 * gyr_scale)),
            }
            extractor.add_reading(livreur_id, reading)

        result.append(extractor.get_features(livreur_id))

    return np.array(result)


def train_and_save(model_path: str = 'model.pkl') -> xgb.Booster:
    print("═" * 50)
    print("  OINSExpress — Entraînement XGBoost")
    print("═" * 50)

    # ── Génération des données synthétiques ──
    # NORMAL     : conduite douce  (échelle 1.0 / 1.0)
    X_normal = simulate_window(acc_scale=1.0, gyr_scale=1.0, n=600)
    y_normal = np.zeros(600, dtype=int)

    # RISKY      : conduite risquée (échelle 2.5 / 3.5)
    X_risky  = simulate_window(acc_scale=2.5, gyr_scale=3.5, n=400)
    y_risky  = np.ones(400, dtype=int)

    # AGGRESSIVE : conduite agressive (échelle 5.0 / 8.0)
    X_aggr   = simulate_window(acc_scale=5.0, gyr_scale=8.0, n=300)
    y_aggr   = np.full(300, 2, dtype=int)

    X = np.vstack([X_normal, X_risky, X_aggr])
    y = np.concatenate([y_normal, y_risky, y_aggr])

    print(f"[DATA] NORMAL={len(y_normal)}  RISKY={len(y_risky)}  AGGRESSIVE={len(y_aggr)}")
    print(f"[DATA] Features : {X.shape[1]}")

    # ── Split train/test (stratifié, sans sklearn) ──
    rng = np.random.default_rng(42)
    indices = np.arange(len(y))
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = indices[y == cls]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * 0.2))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ── Modèle XGBoost (API native, sans sklearn) ──
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        'max_depth':        5,
        'eta':              0.1,
        'subsample':        0.8,
        'colsample_bytree': 0.8,
        'objective':        'multi:softprob',
        'num_class':        3,
        'eval_metric':      'mlogloss',
        'seed':             42,
        'nthread':          4,
    }
    booster = xgb.train(params, dtrain, num_boost_round=150,
                        verbose_eval=False)

    # ── Évaluation (sans sklearn) ──
    proba_matrix = booster.predict(dtest).reshape(-1, 3)
    y_pred = proba_matrix.argmax(axis=1).astype(int)
    acc = (y_pred == y_test).mean()
    print("\n[RÉSULTATS]")
    class_names = ['NORMAL', 'RISKY', 'AGGRESSIVE']
    for i, name in enumerate(class_names):
        tp = ((y_pred == i) & (y_test == i)).sum()
        fp = ((y_pred == i) & (y_test != i)).sum()
        fn = ((y_pred != i) & (y_test == i)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {name:12s}  precision={prec:.2f}  recall={rec:.2f}  f1={f1:.2f}  support={(y_test==i).sum()}")
    print(f"[OK] Accuracy test : {acc:.1%}")

    # ── Sauvegarde ──
    with open(model_path, 'wb') as f:
        pickle.dump(booster, f)
    print(f"[OK] Modèle sauvegardé → {model_path}")
    print("═" * 50)

    return booster


if __name__ == '__main__':
    train_and_save()
