# api/model_handler.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import io
import cv2
import numpy as np
import base64
import os
from pathlib import Path

# 1. Configuration & Constants
NUM_CLASSES = 7
# Weights lookup: $MODEL_PATH, then models/ (Docker image), then frontend/ (versioned copy in the repo)
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    os.getenv("MODEL_PATH"),
    _HERE.parent / "models" / "resnet34_derma_focal_v3.pth",
    _HERE.parent / "frontend" / "resnet34_derma_focal_v3.pth",
]
MODEL_PATH = next((str(p) for p in _CANDIDATES if p and Path(p).exists()), str(_CANDIDATES[1]))

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

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# 2. XAI: Grad-CAM Implementation
# ==========================================
class GradCAM:
    """
    Grad-CAM explains the model's decision by looking at the gradients 
    of the target class flowing into the final convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 1. Register "Hooks" to spy on the layer during Forward and Backward passes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Save the Feature Maps (what the layer "sees")
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # Save the Gradients (how important each Feature Map is for the final answer)
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class_idx):
        """
        Calculates the heatmap for a specific class.
        """
        self.model.eval()
        
        # We need gradients to calculate Grad-CAM, so we CANNOT use torch.no_grad() here
        input_tensor.requires_grad = True
        
        # 1. Forward pass
        model_output = self.model(input_tensor)
        
        # 2. Zero existing gradients, then Backward pass specifically on the target class
        self.model.zero_grad()
        # Create a tensor of 0s, but put a 1 on the class we want to explain
        one_hot_output = torch.zeros((1, model_output.size()[-1]), dtype=torch.float32)
        one_hot_output[0][target_class_idx] = 1
        
        # Trigger the backward pass to calculate gradients
        model_output.backward(gradient=one_hot_output)

        # 3. Calculate Grad-CAM
        # Get the mean of the gradients across the Spatial dimensions (H, W) -> "Weights"
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Multiply each activation map by its corresponding gradient weight
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]
            
        # Sum the channels to create a single heatmap, apply ReLU to keep only positive influence
        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        
        # Normalize the heatmap between 0 and 1
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

def apply_heatmap_to_image(original_image_pil, heatmap_np):
    """
    Overlays the raw heatmap onto the original image.
    Returns the final image as a Base64 encoded string.
    """
    # Resize heatmap to match the original image size (224x224)
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))
    heatmap_resized = np.uint8(255 * heatmap_resized) # Scale to 0-255
    
    # Apply a color map (JET = Red is hot/important, Blue is cold/ignored)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert PIL Image to OpenCV format (BGR instead of RGB)
    original_img_np = np.array(original_image_pil.resize((224, 224)))
    original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
    
    # Blend the heatmap and the original image (0.4 / 0.6 ratio)
    superimposed_img = heatmap_colored * 0.4 + original_img_bgr * 0.6
    
    # Convert back to RGB and then to PIL Image
    final_img_rgb = cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)
    final_pil = Image.fromarray(final_img_rgb)
    
    # Encode the image to Base64 so it can be sent via JSON
    buffered = io.BytesIO()
    final_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

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
        # I. Predict image class and confidence
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
        
        # II. EXPLAINABLE AI (Grad-CAM) - Generate explanation heatmap for the predicted class
        # Initialize Grad-CAM on the last convolutional layer of ResNet34 (layer4)
        grad_cam = GradCAM(model, model.layer4)
        
        # Generate the heatmap for the PREDICTED class
        heatmap = grad_cam.generate_heatmap(tensor, class_idx)
        
        # Overlay heatmap on image and encode to base64
        heatmap_base64 = apply_heatmap_to_image(image, heatmap)

        return {
            "status": "success",
            "prediction": class_name,
            "confidence": round(confidence * 100, 2), # Percentage
            "all_probabilities": all_probs,
            "is_melanoma": (class_idx == 1),
            "heatmap_base64": heatmap_base64
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}