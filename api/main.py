# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_handler import load_model, predict_image

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Lucid Derma-Scan API",
    description="API for detecting melanoma and other skin lesions using ResNet34.",
    version="1.0.0"
)

# 2. Setup CORS (Cross-Origin Resource Sharing)
# Crucial if your frontend (Streamlit/React) runs on a different port than this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the model ONCE when the API starts
# We don't want to load a 100MB model for every single user request!
print("Starting API Server...")
model = load_model()

# 4. Define API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Lucid Derma-Scan API is online and ready."}

@app.post("/predict/")
async def create_prediction(file: UploadFile = File(...)):
    """
    Endpoint that accepts an image file and returns the AI diagnosis.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the file contents as bytes
        image_bytes = await file.read()
        
        # Pass to our PyTorch handler
        result = predict_image(image_bytes, model)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")