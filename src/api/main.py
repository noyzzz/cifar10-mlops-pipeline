# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.ml.infer import load_model, predict_image_bytes

# Global variables for model
_model = None
_classes = None
_device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model once when server starts
    global _model, _classes, _device
    print("Loading CIFAR-10 model...")
    _model, _classes, _device = load_model()
    print(f"Model loaded successfully on {_device}")
    
    yield  # Server runs here
    
    # Shutdown: Cleanup if needed (optional)
    print("Shutting down...")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="CIFAR-10 Classifier API",
    description="Upload an image to get CIFAR-10 classification predictions",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict CIFAR-10 class for uploaded image."""
    # Validate file type
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Please upload an image file (JPEG, PNG, etc.)"
        )
    
    # Read image data
    data = await file.read()
    
    # Make prediction
    try:
        result = predict_image_bytes(_model, _classes, data, _device, topk=10)
        return {
            "filename": file.filename,
            "predictions": result["topk"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "CIFAR-10 Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }
