# frontend/app.py
import streamlit as st
import requests
from PIL import Image
import pandas as pd
import io
import base64  # <-- NEW: Required to decode the heatmap image from the API

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
API_URL = "http://127.0.0.1:8000/predict/"

# ==========================================
# 2. Sidebar (Information & Disclaimer)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004416.png", width=100)
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

if uploaded_file is not None:
    # Create two columns for a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Lesion", use_container_width=True)
        
        analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    with col2:
        st.markdown("### AI Analysis Results")
        
        if analyze_button:
            with st.spinner('Analyzing image and generating Grad-CAM explanation...'):
                try:
                    # Prepare and send the file to the FastAPI backend
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result["status"] == "success":
                            # --- 1. Display Main Prediction ---
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
                            
                            # --- 2. Display Probability Chart ---
                            st.markdown("#### Probability Distribution")
                            probs = result["all_probabilities"]
                            df_probs = pd.DataFrame(list(probs.items()), columns=['Diagnosis', 'Probability'])
                            df_probs = df_probs.sort_values(by='Probability', ascending=True)
                            st.bar_chart(df_probs.set_index('Diagnosis'))
                            
                            # --- 3. Display Explainable AI (Grad-CAM) ---
                            # Check if the heatmap data was sent by the API
                            if "heatmap_base64" in result:
                                st.markdown("---")
                                st.markdown("### 🧠 AI Explanation (Grad-CAM)")
                                st.info("This heatmap reveals how the AI made its decision. **Red zones** indicate the visual features that most strongly influenced the final diagnosis.")
                                
                                # Decode the base64 string back into raw image bytes
                                heatmap_bytes = base64.b64decode(result["heatmap_base64"])
                                # Convert bytes back into a PIL Image
                                heatmap_image = Image.open(io.BytesIO(heatmap_bytes))
                                
                                # Display the heatmap
                                st.image(heatmap_image, caption="Grad-CAM Thermal Map", use_container_width=True)
                                
                        else:
                            st.error(f"API Error: {result.get('message', 'Unknown error')}")
                            
                    else:
                        st.error(f"Server returned status code: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the API. Is the FastAPI server running?")
        else:
            st.info("👈 Click the 'Analyze Image' button to start.")
