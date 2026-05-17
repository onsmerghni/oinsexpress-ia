# Evaluation finale du modele de production

Ce document cloture la partie IA du projet. Il reporte les metriques 
officielles du modele XGBoost de production, evalue sur le test hold-out, 
et detaille la calibration du systeme d'alerte.

## 1. Identification du modele

| Element | Valeur |
|---|---|
| Type | XGBoost Classifier (binaire) |
| Fichier | `models/xgb_driver_behavior.pkl` |
| Classes | AGGRESSIVE, NORMAL |
| Nombre de features | 58 |
| Scaler | `models/scaler.pkl` |
| Label encoder | `models/label_encoder.pkl` |

## 2. Metriques sur le test hold-out

| Metrique | Valeur |
|---|---|
| **Accuracy** | **0.9000** |
| **F1-macro** | **0.8999** |
| F1-weighted | 0.8999 |
| Precision macro | 0.9018 |
| Recall macro | 0.9000 |
| **AUC** | **0.9689** |
| Average Precision | 0.9750 |
| Train Accuracy | 0.9079 |
| **Gap Train-Test** | **+0.0079** (excellent si < 0.10) |

### 2.1 Performance par classe

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| **AGGRESSIVE** | 0.9286 | 0.8667 | 0.8966 |
| **NORMAL** | 0.8750 | 0.9333 | 0.9032 |

### 2.2 Matrice de confusion

```
                  Predit
                  AGGRESSIVE  NORMAL      
Vrai AGGRESSIVE 13          2           
Vrai NORMAL     1           14          
```

**Lecture** : sur 30 echantillons de test, le modele commet 
3 erreurs (1 faux positifs, 2 faux negatifs). 
Le **recall AGGRESSIVE** est de **0.8667**, 
ce qui signifie qu'on detecte 86.7% des vrais cas 
agressifs - critique pour un systeme d'alerte.

## 3. Calibration des seuils d'alerte

Le service Flask declenche **deux niveaux d'alerte** bases sur la probabilite 
P(AGGRESSIVE) renvoyee par le modele :

| Niveau | Seuil | Action |
|---|---|---|
| **WARNING** | P >= 0.50 | Notification simple |
| **CRITICAL** | P >= 0.85 | Alerte au responsable |

### 3.1 Comportement des seuils sur le test

- **14** alertes WARNING declenchees sur 30 echantillons
- **11** d'entre elles montent au niveau CRITICAL
- Precision WARNING  : 13/14 = **92.9%** de vraies alertes
- Precision CRITICAL : 11/11 = **100.0%** de vraies alertes

Voir `final_threshold_calibration.png` pour la visualisation.

## 4. Analyse des erreurs

Le modele commet **3 erreurs** sur 30 predictions :
- **1 faux positifs** (NORMAL classe AGGRESSIVE) - genere de fausses alertes
- **2 faux negatifs** (AGGRESSIVE classe NORMAL) - cas rates (le plus grave)

Voir le detail dans `final_error_analysis.csv`.

## 5. Pipeline de production

Le service Flask `anomaly_detector.py` charge les artefacts suivants au demarrage :
```
models/
  ├── xgb_driver_behavior.pkl    # Modele XGBoost entraine
  ├── scaler.pkl                  # StandardScaler ajuste sur train
  ├── label_encoder.pkl           # NORMAL <-> 1, AGGRESSIVE <-> 0
  ├── feature_cols.pkl            # Liste des 58 features
  └── alert_thresholds.pkl        # Seuils WARNING / CRITICAL
```

**Endpoints REST** :
- `POST /predict` -> recoit les donnees IMU brutes, renvoie {classe, probabilites, niveau_alerte}
- `GET /health`   -> verifie que le service est actif et le modele charge

## 6. Verdict production

**[VALIDE] Le modele est pret pour la production** (criteres reunis) :
- Gap train-test = +0.0079 < 0.10
- Accuracy = 0.9000 >= 0.85
- AUC = 0.9689 >= 0.90

## 7. Limites connues et perspectives

**Limites actuelles** :
- Test set de petite taille (n=30) -> intervalle de confiance large sur les metriques
- Dataset issu de 2 conducteurs uniquement -> generalisation a d'autres profils a confirmer
- Capteur MPU6050 simule via dataset smartphone -> calibration physique a faire en deploiement

**Pistes d'amelioration** :
- Collecter plus de donnees avec le capteur MPU6050 reel monte dans le vehicule
- Ajouter la classe `LOW_NORMAL` (conduite tres lente, possible somnolence)
- Mettre en place un monitoring en production (data drift, performance dans le temps)
- Re-entrainement periodique sur les nouvelles donnees etiquetees

## 8. Fichiers livres

| Fichier | Description |
|---|---|
| `models/xgb_driver_behavior.pkl` | Modele final entraine |
| `models/scaler.pkl` | Normalisation des features |
| `models/label_encoder.pkl` | Encodage des classes |
| `models/feature_cols.pkl` | Schema des 58 features |
| `models/alert_thresholds.pkl` | Seuils WARNING/CRITICAL |
| `reports/final_evaluation_report.json` | Metriques completes |
| `reports/final_confusion_matrix.png` | Matrice de confusion |
| `reports/final_threshold_calibration.png` | Calibration des seuils |
| `reports/final_error_analysis.csv` | Detail de chaque erreur |
| `reports/model_selection_summary.md` | Justification du choix de XGBoost |
| `oinsexpress_ia/flask_service/anomaly_detector.py` | Service de production |

---

*Rapport genere automatiquement par `scripts/08_final_evaluation.py`*