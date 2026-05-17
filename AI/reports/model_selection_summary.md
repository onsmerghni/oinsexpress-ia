# Justification du choix du modele final — XGBoost

**Modele retenu : `XGBoost`**  
**Score composite : 0.8810** 
(2eme = Logistic Regression, ecart +0.1555)

## 1. Contexte du projet

- **Capteur** : MPU6050 (3 axes accelerometre + 3 axes gyroscope)
- **Features** : 58 derivees a partir des 6 signaux bruts (magnitudes, energie, 
  jerk, rolling stats sur fenetre de 10 points)
- **Cible** : classification binaire NORMAL / AGGRESSIVE
- **Systeme d'alerte** : deux seuils probabilistes (WARNING >= 0.50, CRITICAL >= 0.95)
- **Deploiement** : service Flask, inference en temps reel

## 2. Methodologie

Plutot que de se baser sur une seule metrique, nous appliquons une selection 
**multicriteres** combinant 6 axes normalises entre 0 et 1 :

| Critere | Poids | Pourquoi |
|---|---|---|
| F1-macro test | 25% | Performance reelle de generalisation |
| AUC test | 20% | **Critique** : seuils d'alerte probabilistes |
| CV F1-macro | 20% | Robustesse (test n=30 trop petit pour conclure seul) |
| Stabilite CV (1 - std) | 15% | Faible variance entre folds |
| Pas d'overfitting (1 - |gap|) | 15% | Ecart Train-Test faible |
| Vitesse d'inference | 5% | Production temps reel |

## 3. Classement

| Rang | Modele | Score | F1 test | AUC | CV F1 | Gap | Pred (us) |
|---|---|---|---|---|---|---|---|
| #1 | **XGBoost** | 0.8810 | 0.8999 | 0.9733 | 0.8600 | +0.0079 | 24.4 |
| #2 | **Logistic Regression** | 0.7254 | 0.8999 | 0.9556 | 0.8071 | +0.0474 | 4.0 |
| #3 | **Random Forest** | 0.6952 | 0.8661 | 0.9733 | 0.8877 | +0.1333 | 1230.7 |
| #4 | **K-Nearest Neighbors** | 0.5959 | 0.8661 | 0.9600 | 0.8537 | +0.0281 | 482.2 |
| #5 | **SVM (RBF kernel)** | 0.5437 | 0.8331 | 0.9511 | 0.8460 | +0.1272 | 13.9 |
| #6 | **LightGBM** | 0.3467 | 0.7964 | 0.9644 | 0.8614 | +0.1868 | 41.0 |
| #7 | **MLP (neural net)** | 0.2923 | 0.8661 | 0.9422 | 0.6861 | -0.1101 | 3.3 |

## 4. Diagnostic detaille par modele

### XGBoost (score 0.8810)
- [+] Pas d'overfitting (gap=+0.008)
- [+] F1-test eleve (0.900)
- [+] AUC excellent (0.973) - bon pour seuils
- [+] CV stable (std=0.068)

### Logistic Regression (score 0.7254)
- [+] Pas d'overfitting (gap=+0.047)
- [+] F1-test eleve (0.900)
- [~] AUC bon (0.956) - seuils moins fiables
- [+] CV stable (std=0.055)

### Random Forest (score 0.6952)
- [~] Overfitting modere (gap=+0.133)
- [~] F1-test moyen (0.866)
- [+] AUC excellent (0.973) - bon pour seuils
- [+] CV stable (std=0.065)

### K-Nearest Neighbors (score 0.5959)
- [+] Pas d'overfitting (gap=+0.028)
- [~] F1-test moyen (0.866)
- [~] AUC bon (0.960) - seuils moins fiables
- [~] CV moyennement stable (std=0.098)

### SVM (RBF kernel) (score 0.5437)
- [~] Overfitting modere (gap=+0.127)
- [~] F1-test moyen (0.833)
- [~] AUC bon (0.951) - seuils moins fiables
- [+] CV stable (std=0.035)

### LightGBM (score 0.3467)
- [!] OVERFITTING SEVERE (gap=+0.187)
- [!] F1-test faible (0.796)
- [~] AUC bon (0.964) - seuils moins fiables
- [!] CV INSTABLE (std=0.101)

### MLP (neural net) (score 0.2923)
- [~] Overfitting modere (gap=-0.110)
- [~] F1-test moyen (0.866)
- [~] AUC bon (0.942) - seuils moins fiables
- [~] CV moyennement stable (std=0.096)

## 5. Cas particulier : XGBoost vs Logistic Regression

Logistic Regression et XGBoost obtiennent **strictement le meme F1-test** 
(0.8999) et la meme matrice de confusion. 
Cela merite une justification specifique du choix de XGBoost.

### 5.1 Pourquoi le test seul ne suffit pas
- Le test set ne contient que **n=30 echantillons**, ce qui est tres faible. 
- A cette taille, **3 erreurs vs 4 erreurs** donnent des F1 quasi identiques 
  mais ne sont **pas statistiquement significatifs**.
- La **validation croisee** (n=152, 5 folds) est plus fiable pour departager.

### 5.2 La validation croisee departage clairement
| Modele | CV F1 mean | CV F1 std |
|---|---|---|
| **XGBoost** | **0.8600** | 0.0683 |
| Logistic Regression | 0.8071 | 0.0552 |
| **Ecart en faveur de XGBoost** | **+0.0530** | |

Soit **+5.3 points** de F1 
en CV - ecart significatif.

### 5.3 Pourquoi XGBoost est mieux adapte aux donnees MPU6050
Les 58 features derivees du MPU6050 ont trois proprietes qui defavorisent 
Logistic Regression :

1. **Multi-correlation** : `AccMag = sqrt(AccX^2+AccY^2+AccZ^2)`, 
   `AccEnergy = AccMag^2`, `BrakeIntensity = -AccX_roll_min` et 40 rolling 
   stats sur 8 colonnes — beaucoup de redondance. LogReg estime des 
   coefficients **instables** dans ce regime.
2. **Non-linearites** : un virage agressif n'est pas une fonction lineaire 
   de `GyroZ` seul ; il depend de la **combinaison** rotation + acceleration 
   laterale. LogReg suppose une frontiere de decision lineaire.
3. **Interactions** : XGBoost capture naturellement les interactions 
   entre features (ex : *forte GyroZ ET faible AccX_roll_min* = virage serre 
   avec freinage). LogReg ne le fait pas sans features croisees manuelles.

### 5.4 L'AUC est critique pour le systeme d'alerte
Notre architecture utilise **deux seuils probabilistes** :
- `SEUIL_WARNING = 0.50`
- `SEUIL_CRITICAL = 0.95`

La qualite de **separation probabiliste** (AUC) est donc critique.

- **XGBoost AUC = 0.9733** 
- Logistic Regression AUC = 0.9556 
(ecart -0.0178)

Un AUC plus eleve signifie que les probabilites P(AGGRESSIVE) sont **mieux 
calibrees** pour distinguer les cas. Les seuils d'alerte deviennent plus 
fiables : moins de faux WARNINGs et moins de vrais CRITICAL rates.

### 5.5 Conclusion de la comparaison
| Critere | Gagnant | Importance |
|---|---|---|
| F1 test (n=30) | Egalite | Peu fiable a cette taille |
| **CV F1 (n=152)** | **XGBoost (+5.3 pts)** | **Plus fiable** |
| **AUC** | **XGBoost (+1.8 pts)** | **Critique pour seuils** |
| Adequation aux features MPU6050 | XGBoost | Non-linearites + interactions |
| Simplicite (Occam) | LogReg | Argument secondaire |
| Vitesse | LogReg (6x) | Negligeable (24 us vs 4 us, les deux < 1 ms) |

## 6. Conclusion finale

**`XGBoost` est le modele de production retenu**.

Ce choix repose sur une analyse multicriteres rigoureuse qui tient compte 
des specificites du projet (donnees MPU6050 multi-correlees et non-lineaires, 
systeme d'alerte probabiliste a deux seuils, contraintes temps reel). 
Aucun autre modele ne reunit simultanement performance, robustesse, 
absence d'overfitting et qualite probabiliste.

**Voir aussi** : 
- `model_selection_ranking.png` (classement visuel)
- `model_selection_radar.png` (profil multicriteres top 4)
- `benchmark_results.csv` (donnees brutes)
- `final_evaluation_report.json` (re-evaluation du modele final)