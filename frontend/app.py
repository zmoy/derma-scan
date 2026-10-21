# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import io
import os
import random

# ==========================================
# 1. Page Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="Lucid Derma-Scan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Sample Images ---
SAMPLE_DIR = "sample_images"
MODEL_PATH = "resnet34_derma_focal_v3.pth"

MELANOMA_SAMPLES = [
    "ISIC_0032511.jpg", "ISIC_0032812.jpg", "ISIC_0034012.jpg", 
    "ISIC_0034061.jpg", "ISIC_0032547.jpg"
]
BENIGN_SAMPLES = [
    "ISIC_0030321.jpg", "ISIC_0031503.jpg", "ISIC_0033620.jpg", 
    "ISIC_0031191.jpg", "ISIC_0025321.jpg", "ISIC_0027107.jpg"
]

# Class order used at training time (HAM10000 label_map: nv, mel, bkl, bcc, akiec, vasc, df).
# It MUST match the output order of the trained model.
CLASSES = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis', 'Basal cell carcinoma',
           'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']
BENIGN_CLASSES = {'Melanocytic nevi', 'Benign keratosis', 'Vascular lesions', 'Dermatofibroma'}

# ==========================================
# 2. AI Model Loading & Processing Logic
# ==========================================

# @st.cache_resource is CRITICAL here. It ensures PyTorch only loads the 
# model once into RAM, preventing Out-Of-Memory crashes on cloud servers.
@st.cache_resource
def load_model():
    # Define the architecture exactly as it was trained
    model = models.resnet34(weights=None) # We don't need pretrained weights from internet, we have ours
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    # Load your fine-tuned weights (forced to CPU for cloud free tiers)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Same preprocessing as validation time (ImageNet statistics)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_gradcam(image_tensor, model, target_layer):
    """
    Minimal Grad-CAM implementation.
    """
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Remove hooks
    hook_f.remove()
    hook_b.remove()

    # Calculate heatmap
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations[0].shape[1]):
        activations[0][:, i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations[0], dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.detach().numpy(), output

# ==========================================
# 3. Sidebar (Information & Disclaimer)
# ==========================================
with st.sidebar:
    st.image("logo.png", width=100)
    st.title("About Derma-Scan")
    st.write("This AI tool analyzes dermatological images to classify 7 different types of skin lesions using a fine-tuned ResNet34 model.")
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("**Not a substitute for professional medical advice.** "
        "This application is a Proof of Concept (PoC) for educational and portfolio purposes only. "
        "Any suspicious skin lesion should be examined by a certified dermatologist.")
    st.markdown("---")
    st.write("Developed by: Zacharie MOŸ")

# ==========================================
# 4. Main Interface
# ==========================================
st.title("🔬 Lucid Derma-Scan AI")
st.subheader("Skin Lesion Classification Tool")
st.write("Upload a high-resolution, close-up image of a skin mole or lesion to get an AI-powered preliminary diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.markdown("**Don't have an image? Try a curated sample:**")
col_s1, col_s2 = st.columns(2)
with col_s1:
    test_mel = st.button("🚨 Quick Test: Melanoma Case", width="stretch")
with col_s2:
    test_ben = st.button("✅ Quick Test: Benign Case", width="stretch")

image_source = None
file_name = ""
auto_analyze = False

if test_mel:
    file_name = random.choice(MELANOMA_SAMPLES)
    image_source = os.path.join(SAMPLE_DIR, file_name)
    auto_analyze = True
elif test_ben:
    file_name = random.choice(BENIGN_SAMPLES)
    image_source = os.path.join(SAMPLE_DIR, file_name)
    auto_analyze = True
elif uploaded_file is not None:
    image_source = uploaded_file
    file_name = uploaded_file.name

if image_source is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        # Ensure we load a clean PIL image
        if isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        else:
            image_source.seek(0)
            image = Image.open(image_source).convert("RGB")
            
        st.image(image, caption=f"Lesion: {file_name}", width="stretch")
        analyze_button = st.button("🔍 Analyze Image", type="primary", width="stretch")

    with col2:
        st.markdown("### AI Analysis Results")
        
        if analyze_button or auto_analyze:
            with st.spinner('Analyzing image and generating Grad-CAM explanation...'):
                # 1. Preprocess the image
                input_tensor = transform(image).unsqueeze(0)
                
                # 2. Run Grad-CAM and get predictions
                target_layer = model.layer4[-1].conv2 # last conv layer of ResNet34
                heatmap_arr, outputs = generate_gradcam(input_tensor, model, target_layer)
                
                # 3. Calculate probabilities
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidences = {CLASSES[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(CLASSES))}
                
                pred_class_idx = torch.argmax(probabilities).item()
                pred_class = CLASSES[pred_class_idx]
                confidence = confidences[pred_class]
                is_melanoma = (pred_class == 'Melanoma')

                # --- Display Main Prediction ---
                if is_melanoma:
                    st.error(f"🚨 **CRITICAL ALERT: {pred_class}**")
                    st.error(f"Confidence: **{confidence}%**")
                elif pred_class in BENIGN_CLASSES:
                    st.success(f"✅ **Prediction: {pred_class}**")
                    st.success(f"Confidence: **{confidence}%**")
                else:
                    st.warning(f"⚠️ **Prediction: {pred_class}**")
                    st.warning(f"Confidence: **{confidence}%**")
                
                # --- Display Probability Chart ---
                st.markdown("#### Probability Distribution")
                df_probs = pd.DataFrame(list(confidences.items()), columns=['Diagnosis', 'Probability'])
                df_probs = df_probs.sort_values(by='Probability', ascending=True)
                st.bar_chart(df_probs.set_index('Diagnosis'))
                
                # --- Display Explainable AI (Grad-CAM) ---
                st.markdown("---")
                st.markdown("### 🧠 AI Explanation (Grad-CAM)")
                st.info("This heatmap reveals how the AI made its decision. **Red zones** indicate the visual features that most strongly influenced the final diagnosis.")
                
                # Overlay heatmap on original image
                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                heatmap_resized = cv2.resize(heatmap_arr, (img_cv.shape[1], img_cv.shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
                superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
                
                st.image(superimposed_img, caption="Grad-CAM Thermal Map", width="stretch")