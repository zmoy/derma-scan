# api/model_handler.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import io

# 1. Configuration & Constants
NUM_CLASSES = 7
MODEL_PATH = "../models/resnet34_derma_focal_v3.pth" # Path from api/ to models/
# MODEL_PATH = "models/resnet34_derma_focal_v3.pth" # Path from api/ to models/

# Label mapping for human-readable output
LABEL_MAP = {
    0: 'Melanocytic nevi (Benign)',
    1: 'Melanoma (CRITICAL)',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma (Danger)',
    4: 'Actinic keratoses (Pre-cancerous)',
    5: 'Vascular lesions (Benign)',
    6: 'Dermatofibroma (Benign)'
}

# 2. Image Transformations
# These MUST exactly match the transforms used during validation/testing
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# 3. Model Loading Strategy
def load_model():
    """
    Initializes ResNet34, replaces the head, and loads the fine-tuned weights.
    """
    print("Loading PyTorch model into memory...")
    # Initialize base architecture
    model = models.resnet34(weights=None) # We don't need ImageNet weights anymore
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load our trained weights
    # map_location=torch.device('cpu') is crucial if the server running the API doesn't have a GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    model.eval() # Set to evaluation mode (disables dropout, etc.)
    print("Model loaded successfully.")
    return model

def predict_image(image_bytes: bytes, model: nn.Module):
    """
    Takes raw image bytes, runs them through the model, and returns a prediction dictionary.
    """
    try:
        # 1. Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 2. Apply transformations and add batch dimension (C,H,W -> 1,C,H,W)
        tensor = image_transforms(image).unsqueeze(0)
        
        # 3. Forward pass (Inference)
        with torch.no_grad():
            outputs = model(tensor)
            # Apply Softmax to get probabilities between 0 and 1
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 4. Extract results
        top_prob, top_class_idx = torch.max(probabilities, dim=0)
        
        # Convert tensors to standard Python types for JSON serialization
        confidence = top_prob.item()
        class_idx = top_class_idx.item()
        class_name = LABEL_MAP[class_idx]
        
        # Create a dictionary of all probabilities for the front-end bar chart
        all_probs = {LABEL_MAP[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return {
            "status": "success",
            "prediction": class_name,
            "confidence": round(confidence * 100, 2), # Percentage
            "all_probabilities": all_probs,
            "is_melanoma": (class_idx == 1)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}