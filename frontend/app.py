# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io
import base64 
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

# API Endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict/")

# --- Constants for Sample Images ---
SAMPLE_DIR = "sample_images"

MELANOMA_SAMPLES = [
    "ISIC_0032511.jpg", "ISIC_0032812.jpg", "ISIC_0034012.jpg", 
    "ISIC_0034061.jpg", "ISIC_0032547.jpg"
]
BENIGN_SAMPLES = [
    "ISIC_0030321.jpg", "ISIC_0031503.jpg", "ISIC_0033620.jpg", 
    "ISIC_0031191.jpg", "ISIC_0025321.jpg", "ISIC_0027107.jpg"
]

# ==========================================
# 2. Sidebar (Information & Disclaimer)
# ==========================================
with st.sidebar:
    st.image("logo.png", width=100)
    st.title("About Derma-Scan")
    st.write("This AI tool analyzes dermatological images to classify 7 different types of skin lesions using a fine-tuned ResNet34 model.")
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning(
        "**Not a substitute for professional medical advice.** "
        "This application is a Proof of Concept (PoC) for educational and portfolio purposes only. "
        "Any suspicious skin lesion should be examined by a certified dermatologist."
    )
    st.markdown("---")
    st.write("Developed by: Zacharie MOŸ")

# ==========================================
# 3. Main Interface
# ==========================================
st.title("🔬 Lucid Derma-Scan AI")
st.subheader("Skin Lesion Classification Tool")
st.write("Upload a high-resolution, close-up image of a skin mole or lesion to get an AI-powered preliminary diagnosis.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- One-Click Sample Test Buttons ---
st.markdown("**Don't have an image? Try a curated sample:**")
col_s1, col_s2 = st.columns(2)
with col_s1:
    test_mel = st.button("🚨 Quick Test: Melanoma Case", width="stretch")
with col_s2:
    test_ben = st.button("✅ Quick Test: Benign Case", width="stretch")

# ==========================================
# 4. Image Processing Logic
# ==========================================
# Determine the source of the image (User upload vs. Sample button)
image_source = None
file_name = ""
mime_type = "image/jpeg"
auto_analyze = False # Flag to skip the "Analyze" button click if a sample is chosen

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
    mime_type = uploaded_file.type

# If an image has been provided (either way), display the analysis UI
if image_source is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        
        # Load image depending on its source (File path vs. BytesIO from uploader)
        if isinstance(image_source, str):
            # It's a local sample file
            image = Image.open(image_source)
            with open(image_source, "rb") as f:
                image_bytes = f.read()
        else:
            # It's a user-uploaded file
            image = Image.open(image_source)
            image_source.seek(0)
            image_bytes = image_source.read()
            
        st.image(image, caption=f"Lesion: {file_name}", width="stretch")
        
        # Display the analyze button (only strictly necessary if user uploaded a file)
        analyze_button = st.button("🔍 Analyze Image", type="primary", width="stretch")

    with col2:
        st.markdown("### AI Analysis Results")
        
        # Trigger analysis if user clicked 'Analyze' OR if a sample button was pressed
        if analyze_button or auto_analyze:
            with st.spinner('Analyzing image and generating Grad-CAM explanation...'):
                try:
                    # Send bytes directly to FastAPI
                    files = {"file": (file_name, image_bytes, mime_type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result["status"] == "success":
                            # --- Display Main Prediction ---
                            pred_class = result["prediction"]
                            confidence = result["confidence"]
                            
                            if result["is_melanoma"]:
                                st.error(f"🚨 **CRITICAL ALERT: {pred_class}**")
                                st.error(f"Confidence: **{confidence}%**")
                            elif "Benign" in pred_class:
                                st.success(f"✅ **Prediction: {pred_class}**")
                                st.success(f"Confidence: **{confidence}%**")
                            else:
                                st.warning(f"⚠️ **Prediction: {pred_class}**")
                                st.warning(f"Confidence: **{confidence}%**")
                            
                            # --- Display Probability Chart ---
                            st.markdown("#### Probability Distribution")
                            probs = result["all_probabilities"]
                            df_probs = pd.DataFrame(list(probs.items()), columns=['Diagnosis', 'Probability'])
                            df_probs = df_probs.sort_values(by='Probability', ascending=True)
                            st.bar_chart(df_probs.set_index('Diagnosis'))
                            
                            # --- Display Explainable AI (Grad-CAM) ---
                            if "heatmap_base64" in result:
                                st.markdown("---")
                                st.markdown("### 🧠 AI Explanation (Grad-CAM)")
                                st.info("This heatmap reveals how the AI made its decision. **Red zones** indicate the visual features that most strongly influenced the final diagnosis.")
                                
                                heatmap_bytes = base64.b64decode(result["heatmap_base64"])
                                heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                                st.image(heatmap_image, caption="Grad-CAM Thermal Map", width="stretch")
                                
                        else:
                            st.error(f"API Error: {result.get('message', 'Unknown error')}")
                            
                    else:
                        st.error(f"Server returned status code: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API. Is the FastAPI server running?")