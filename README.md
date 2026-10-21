# 🔬 Lucid Derma-Scan AI: Dermatological Triage Tool

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg)

## 🌐 Live Demo
Experience the application instantly without any local setup. A streamlined, monolithic version of this project has been deployed on Streamlit Community Cloud for public demonstration.

👉 **[Try Lucid Derma-Scan AI Live Here](https://lucid-derma-scan.streamlit.app/)**

## 📖 Project Overview
**Lucid Derma-Scan** is a Full-Stack Artificial Intelligence Proof of Concept (PoC) designed for the medical field. Its goal is to act as a **triage tool** to help detect melanoma and classify 6 other types of skin lesions from dermoscopic images.

The application is divided into microservices: an optimized PyTorch model, a robust backend API (FastAPI), and an interactive user interface (Streamlit) integrating Explainable AI (Grad-CAM).

## 🧠 Modeling and Machine Learning
The main challenge of this project was the **severe class imbalance** (melanoma being a rare class compared to healthy moles). 

### Model Evolution:
1. **Naive Model (Baseline):** A standard model heavily favored the majority class, resulting in a catastrophic 16% recall for melanoma. Unacceptable in a medical context.
2. **Rebalancing Strategy:** Implemented a `WeightedRandomSampler` to force the model to see rare classes, and created a custom **Focal Loss** function to heavily penalize errors on critical cases.
3. **Architecture and Fine-Tuning:** After a comparative benchmark with EfficientNet-B2 and DenseNet121, the strategic choice fell on a **deeply fine-tuned ResNet34** (unfreezing `layer4` with differential Learning Rates).
   * **Final Result:** Recall for melanoma detection jumped from 16% to **82%** on the standard validation split (**71%** once lesion-level data leakage is removed, see below), making this model a usable first-filter tool with fast inference times.

### 📊 Rigorous Evaluation
Recall alone is not enough in a triage setting, so the final model was evaluated with a full metric suite (`evaluate_model.py`): per-class precision/recall/specificity/NPV, balanced accuracy, macro-F1, MCC, ROC/PR-AUC, calibration (ECE), a melanoma threshold sweep and bootstrap 95% confidence intervals.

| Metric (validation set) | Image-level split | Leakage-free (lesion-level) |
|---|---|---|
| Images | 2003 | 1212 |
| Melanoma recall | 82.1% | 71.0% (95% CI 59.5-81.0) |
| Melanoma precision | 33.2% | 24.7% |
| Melanoma NPV | 97.2% | 98.0% |
| Melanoma ROC-AUC | 0.883 | 0.883 |
| Balanced accuracy | 75.7% | 72.4% |
| Macro-F1 | 0.587 | 0.519 |
| Macro ROC-AUC (OvR) | 0.950 | 0.954 |

**Methodological note.** HAM10000 contains several images of the same lesion (`lesion_id`). The original 80/20 split was made per image, so ~39% of the validation images shared a lesion with the training set. Re-evaluating on the subset of validation images whose lesion is *not* in the training set gives the more honest figures above. A lesion-grouped split (`python evaluate_model.py --export-grouped-split`) is provided for retraining.

**Honest take:** the model ranks melanomas well (AUC 0.88) and rarely misses when it says "not melanoma" (NPV ~98%), but its precision is low (~1 alert in 3-4 is a true melanoma), which is the expected cost of optimising sensitivity. Lowering the decision threshold trades specificity for recall (e.g. 84% recall at threshold 0.2 on the leakage-free set).

Full reports: [`evaluation/report.md`](evaluation/report.md) and [`evaluation_lesion_clean/report.md`](evaluation_lesion_clean/report.md).

```bash
python evaluate_model.py --split lesion-clean --weights frontend/resnet34_derma_focal_v3.pth
```
*(requires the HAM10000 dataset in `data/`, not included in the repo.)*

## 👁️ Explainable AI (XAI)
In the medical field, a "black box" is unacceptable. This project integrates **Grad-CAM** (Gradient-weighted Class Activation Mapping). 
When an image is analyzed, the API intercepts the gradients of the final convolutional layer of the ResNet34 to generate a **thermal heatmap**. This allows the doctor to see exactly *which areas of the lesion* drove the model's decision (e.g., irregular borders, asymmetry).

## ⚙️ Technical Architecture
The project is fully containerized to ensure reproducibility:
* **Backend:** `FastAPI` (PyTorch inference, Grad-CAM generation, Base64 encoding).
* **Frontend:** `Streamlit` (UI/UX interface, asynchronous HTTP requests).
* **Deployment:** `Docker` & `Docker-Compose` (Internal bridge network, build optimization with CPU-only PyTorch).

---

## 🚀 How to Run the Project Locally

Thanks to Docker, launching the complete architecture requires only a single command.

**Prerequisites:** Have [Docker](https://www.docker.com/) and Docker-Compose installed on your machine.

1. Clone this repository:
   ```bash
   git clone https://github.com/zmoy/derma-scan-assistant.git
   cd derma-scan-assistant
   ```

2. Build and run the containers:
   ```bash
   docker-compose up --build
   ```

   The fine-tuned weights are versioned in `frontend/resnet34_derma_focal_v3.pth` (the `models/` folder is git-ignored and only holds local training checkpoints).

3. Access the web app:
   Open your browser and navigate to http://localhost:8501

⚡ One-Click Quick Test
Don't have a dermoscopic image on hand to test the application? No problem.
The web interface includes a built-in Quick Test feature. Simply click the "Melanoma Case" or "Benign Case" buttons, and the app will automatically select a curated sample image from the dataset and run the full end-to-end analysis instantly.

(The FastAPI backend is accessible in parallel at http://localhost:8000/docs to directly test the endpoints).

# ⚠️ Medical Disclaimer
This project is developed exclusively for educational and technical demonstration (Portfolio) purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Any suspicious lesion must be examined by a certified dermatologist.

# Developed with passion by Zacharie MOŸ