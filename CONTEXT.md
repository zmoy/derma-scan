# CONTEXT — Lucid Derma-Scan AI

Document de contexte pour reprendre le projet rapidement (humain ou agent).

## 1. Résumé

PoC de **triage dermatologique** : classification de lésions cutanées à partir d'images dermoscopiques (dataset **HAM10000 / ISIC**, 7 classes), avec explicabilité **Grad-CAM**. Objectif clé : maximiser le **recall du mélanome** (82 % obtenu, contre 16 % pour le modèle naïf). Projet portfolio, pas un dispositif médical (disclaimer présent dans l'UI et le README).

Auteur : Zacharie MOŸ. Démo en ligne : https://lucid-derma-scan.streamlit.app/ (version monolithique).

## 2. Deux modes de déploiement

| Mode | Fichiers | Description |
|---|---|---|
| **Microservices (Docker)** | `api/`, `frontend/`, `docker-compose.yml` | FastAPI (inférence + Grad-CAM) + Streamlit qui appelle l'API en HTTP |
| **Monolithe (Streamlit Cloud)** | `frontend/app.py` (version actuelle) | Streamlit charge le modèle PyTorch directement, sans API |

`frontend/app.py` est la version monolithique active (l'ancien client API en commentaire a été supprimé, il reste dans l'historique git). L'API FastAPI (`api/`) reste fonctionnelle et testable seule ; le service `frontend` de `docker-compose.yml` n'en dépend plus.

## 3. Structure du dépôt

```
api/
  main.py            FastAPI : GET / (health), POST /predict/ (upload image), CORS *
  model_handler.py   load_model, predict_image, classe GradCAM, overlay base64
  Dockerfile         python:3.10-slim, copie api/ et models/, uvicorn :8000
  requirements.txt   torch CPU (extra-index-url), fastapi, uvicorn, opencv-headless...
frontend/
  app.py             UI Streamlit (voir §2)
  Dockerfile         python:3.10-slim, streamlit :8501
  requirements.txt   streamlit, torch, torchvision, Pillow, pandas, numpy, opencv-python-headless
  sample_images/     11 images ISIC pour les boutons « Quick Test »
  logo.png
  resnet34_derma_focal_v3.pth   (non suivi par git, copie du modèle pour la version monolithique)
models/              poids .pth (gitignored sauf v3 historiquement commité)
data/                HAM10000_metadata.csv + all_images/ (gitignored)
notebooks/
  01_Exploratory_Data_Analysis   EDA, déséquilibre de classes
  02_Data_Preprocessing          mapping des labels, splits
  03_Model_Building              ResNet34, WeightedRandomSampler, Focal Loss, fine-tuning
  04_Model_Benchmarking          comparaison EfficientNet-B2 / DenseNet121 / ResNet34
  utils.py                       boucle d'entraînement (scheduler + early stopping), métriques
docker-compose.yml   services api (8000) et frontend (8501), réseau bridge derma-network
requirements.txt     dépendances de dev/notebooks (jupyter, sklearn, seaborn...)
app/                 dossier vide
venv/                environnement virtuel local
```

## 4. Modèle

- **ResNet34** (torchvision), `fc` remplacée par `Linear(512, 7)`.
- Entraînement : `WeightedRandomSampler` + **Focal Loss**, dégel de `layer4` avec learning rates différenciés, scheduler + early stopping.
- Poids de production : `resnet34_derma_focal_v3.pth` (~85 Mo). Autres poids dans `models/` : v1, finetuned_v2, benchmarks DenseNet121 / EfficientNet-B2.
- Prétraitement : Resize 224×224, ToTensor, Normalize ImageNet (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`).
- Inférence CPU (`map_location='cpu'`).

### Ordre des classes (d'entraînement, notebooks 02/03)

`nv=0, mel=1, bkl=2, bcc=3, akiec=4, vasc=5, df=6`

## 5. Grad-CAM

Hooks sur la dernière couche convolutive de ResNet34 → gradients poolés → pondération des activations → ReLU → normalisation → `COLORMAP_JET` superposé à l'image (0.6 image / 0.4 heatmap).
- API : cible `model.layer4`, retourne le JPEG en base64 (`heatmap_base64`), taille 224×224.
- Monolithe : cible `model.layer4[-1].conv2`, overlay à la taille de l'image d'origine.

## 6. API

`POST /predict/` (multipart `file`, doit être `image/*`) → JSON :
`status`, `prediction`, `confidence` (%), `all_probabilities` (0-1), `is_melanoma` (index 1), `heatmap_base64`.
Docs Swagger : http://localhost:8000/docs.

## 7. Lancer le projet

```bash
docker-compose up --build     # UI http://localhost:8501, API http://localhost:8000/docs
```
Local sans Docker : `pip install -r api/requirements.txt`, puis `cd api && uvicorn main:app` (le chemin du modèle est relatif : `../models/...`) ; `cd frontend && streamlit run app.py`.

## 8. Évaluation (`evaluate_model.py`)

- `python evaluate_model.py` : split d'origine par image → `evaluation/`.
- `--split lesion-clean` : retire les images de validation dont le `lesion_id` est dans le train (39,5 % de fuite) → `evaluation_lesion_clean/`. Rappel mélanome réel ≈ **71 %** (82 % avec la fuite).
- `--export-grouped-split` : écrit `grouped_split.csv` (split groupé par `lesion_id`) pour **réentraîner** (non fait à ce jour).
- Seuls les `report.md` sont versionnés ; JSON/PNG/CSV sont ignorés.

## 9. Points d'attention / dette technique

1. Le modèle n'a jamais été réentraîné sur un split groupé par lésion ; les métriques « propres » viennent d'un sous-ensemble de validation (1212 images, 69 mélanomes).
2. `models/` est gitignoré (poids d'entraînement locaux) ; les poids de prod sont versionnés dans `frontend/resnet34_derma_focal_v3.pth` (requis par Streamlit Cloud et par `api/Dockerfile`). Une ancienne copie reste dans l'historique git.
3. `generate_gradcam` du monolithe utilise `register_backward_hook` (déprécié) ; dans l'API, les hooks de `GradCAM` ne sont jamais retirés (une instance par requête sur le modèle partagé).
4. `use_container_width` dans le monolithe (API Streamlit selon la version).
5. CORS `*` dans l'API (dev uniquement), clé `version` obsolète dans `docker-compose.yml`, `requirements.txt` racine non épinglé, dossier `app/` vide.
6. Les « Quick Test » piochent au hasard dans deux listes fixes d'images ISIC (5 mélanomes, 6 bénins), vérifiées avec le modèle.
