# Évaluation Derma-Scan — `frontend/resnet34_derma_focal_v3.pth`

Jeu de validation : 2003 images (split stratifié 80/20, seed 42).

## Métriques globales

| Métrique | Valeur |
|---|---|
| Accuracy | 0.660 |
| Balanced accuracy | 0.757 [0.724 – 0.792] |
| F1 macro | 0.587 [0.547 – 0.619] |
| F1 pondéré | 0.692 |
| MCC | 0.531 [0.503 – 0.554] |
| Cohen's kappa | 0.489 |
| Log-loss | 0.865 |
| ROC-AUC macro (OvR) | 0.950 |
| PR-AUC macro | 0.741 |
| ECE (calibration) | 0.094 |

## Par classe

| Classe | Support | Rappel | Spécificité | Précision | NPV | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| nv | 1341 | 0.620 | 0.976 | 0.981 | 0.559 | 0.760 | 0.938 | 0.970 |
| mel | 223 | 0.821 | 0.793 | 0.332 | 0.972 | 0.473 | 0.883 | 0.513 |
| bkl | 220 | 0.645 | 0.941 | 0.573 | 0.956 | 0.607 | 0.908 | 0.668 |
| bcc | 103 | 0.699 | 0.975 | 0.600 | 0.984 | 0.646 | 0.967 | 0.719 |
| akiec | 65 | 0.738 | 0.964 | 0.410 | 0.991 | 0.527 | 0.966 | 0.651 |
| vasc | 28 | 0.821 | 0.991 | 0.561 | 0.997 | 0.667 | 0.996 | 0.843 |
| df | 23 | 0.957 | 0.971 | 0.278 | 0.999 | 0.431 | 0.993 | 0.825 |

## Mélanome

- Rappel (sensibilité) : 0.821 [0.770 – 0.865]
- Spécificité : 0.793 — Précision (VPP) : 0.332 — NPV : 0.972
- ROC-AUC : 0.883 [0.859 – 0.904] — PR-AUC : 0.513
- TP=183 FN=40 FP=368 TN=1412
- Mélanomes ratés → classés comme : {'nv': 7, 'bkl': 7, 'bcc': 3, 'akiec': 15, 'vasc': 1, 'df': 7}

### Balayage de seuil (proba 'mel' ≥ t)

| Seuil | Sensibilité | Spécificité | Précision | FN | FP |
|---|---|---|---|---|---|
| 0.1 | 0.969 | 0.533 | 0.206 | 7 | 832 |
| 0.2 | 0.915 | 0.640 | 0.242 | 19 | 640 |
| 0.3 | 0.892 | 0.717 | 0.283 | 24 | 504 |
| 0.4 | 0.830 | 0.776 | 0.317 | 38 | 398 |
| 0.5 | 0.749 | 0.837 | 0.365 | 56 | 290 |

## Tâche clinique : malin (mel + bcc + akiec) vs bénin

- Sensibilité : 0.903 — Spécificité : 0.730
- Précision : 0.448 — NPV : 0.969 — F1 : 0.599
- ROC-AUC : 0.903 — PR-AUC : 0.723

## Limites

- Le jeu de validation a servi à la sélection du modèle (early stopping / meilleur checkpoint) : les chiffres sont légèrement optimistes, pas un vrai jeu de test.
- Le split est fait **par image** ; HAM10000 contient plusieurs images par lésion (`lesion_id`), ce qui peut provoquer une fuite entre train et validation.
- Dataset mono-source (ISIC) : aucune validation externe ni sur images de smartphone.
- Les IC bootstrap ne reflètent que la variance d'échantillonnage (peu d'exemples pour df/vasc/akiec).
