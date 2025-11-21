# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from src.ml.infer import load_model, predict_image_bytes
from src.ml.infer_batch import infer_batch
import os
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


@app.post("/predict_batch")
async def predict_batch(file_path: str, subset: Optional[int] = None):
    """Predict on a CIFAR batch pickle file provided by path.

    Expects `file_path` to be a filesystem path to a CIFAR batch (pickle),
    e.g. `data/cifar-10-batches-py/test_batch`.
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="Please provide a dataset file path")

    # Validate file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"Batch file not found: {file_path}")

    # Validate subset if provided
    if subset is not None and subset <= 0:
        raise HTTPException(status_code=400, detail="subset must be a positive integer")

    # Call the batch inference helper which returns a dict with results
    try:
        # We pass only the batch path; `infer_batch` will load the model itself
        result = infer_batch(batch_path=file_path, topk=10, subset=subset)
        return {
            "batch": result.get("batch"),
            "count": result.get("count"),
            "results": result.get("results"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



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
