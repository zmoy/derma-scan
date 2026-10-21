# Évaluation Derma-Scan — `frontend/resnet34_derma_focal_v3.pth`

Jeu de validation : 1212 images. Split d'origine stratifié 80/20 (seed 42) ; 791/2003 (39.5%) des images de validation partagent un `lesion_id` avec le train. **Mode lesion-clean : ces images sont retirées**, il reste 1212 images ({'nv': 951, 'bkl': 104, 'mel': 69, 'bcc': 35, 'akiec': 33, 'vasc': 14, 'df': 6}). Effectifs faibles pour mel/df/vasc : lire les IC avec prudence.

## Métriques globales

| Métrique | Valeur |
|---|---|
| Accuracy | 0.748 |
| Balanced accuracy | 0.724 [0.668 – 0.777] |
| F1 macro | 0.519 [0.465 – 0.569] |
| F1 pondéré | 0.789 |
| MCC | 0.533 [0.494 – 0.566] |
| Cohen's kappa | 0.497 |
| Log-loss | 0.655 |
| ROC-AUC macro (OvR) | 0.954 |
| PR-AUC macro | 0.665 |
| ECE (calibration) | 0.047 |

## Par classe

| Classe | Support | Rappel | Spécificité | Précision | NPV | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| nv | 951 | 0.770 | 0.973 | 0.991 | 0.537 | 0.866 | 0.967 | 0.991 |
| mel | 69 | 0.710 | 0.870 | 0.247 | 0.980 | 0.367 | 0.883 | 0.377 |
| bkl | 104 | 0.625 | 0.963 | 0.613 | 0.965 | 0.619 | 0.912 | 0.655 |
| bcc | 35 | 0.743 | 0.977 | 0.491 | 0.992 | 0.591 | 0.971 | 0.679 |
| akiec | 33 | 0.576 | 0.969 | 0.339 | 0.988 | 0.427 | 0.957 | 0.462 |
| vasc | 14 | 0.643 | 0.988 | 0.391 | 0.996 | 0.486 | 0.993 | 0.758 |
| df | 6 | 1.000 | 0.974 | 0.162 | 1.000 | 0.279 | 0.996 | 0.737 |

## Mélanome

- Rappel (sensibilité) : 0.710 [0.595 – 0.810]
- Spécificité : 0.870 — Précision (VPP) : 0.247 — NPV : 0.980
- ROC-AUC : 0.883 [0.840 – 0.921] — PR-AUC : 0.377
- TP=49 FN=20 FP=149 TN=994
- Mélanomes ratés → classés comme : {'nv': 4, 'bkl': 4, 'bcc': 2, 'akiec': 8, 'vasc': 1, 'df': 1}

### Balayage de seuil (proba 'mel' ≥ t)

| Seuil | Sensibilité | Spécificité | Précision | FN | FP |
|---|---|---|---|---|---|
| 0.1 | 0.942 | 0.653 | 0.141 | 4 | 397 |
| 0.2 | 0.841 | 0.757 | 0.173 | 11 | 278 |
| 0.3 | 0.841 | 0.822 | 0.221 | 11 | 204 |
| 0.4 | 0.725 | 0.862 | 0.240 | 19 | 158 |
| 0.5 | 0.623 | 0.897 | 0.267 | 26 | 118 |

## Tâche clinique : malin (mel + bcc + akiec) vs bénin

- Sensibilité : 0.891 — Spécificité : 0.828
- Précision : 0.397 — NPV : 0.983 — F1 : 0.550
- ROC-AUC : 0.925 — PR-AUC : 0.675

## Limites

- Le jeu de validation a servi à la sélection du modèle (early stopping / meilleur checkpoint) : les chiffres sont légèrement optimistes, pas un vrai jeu de test.
- Si le split est fait **par image** ; HAM10000 contient plusieurs images par lésion (`lesion_id`), ce qui provoque une fuite entre train et validation (voir le mode `lesion-clean`).
- Dataset mono-source (ISIC) : aucune validation externe ni sur images de smartphone.
- Les IC bootstrap ne reflètent que la variance d'échantillonnage (peu d'exemples pour df/vasc/akiec).
