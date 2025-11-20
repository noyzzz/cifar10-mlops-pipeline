# 🎓 CIFAR-10 MLOps Workshop

Learn MLOps by building a complete machine learning pipeline from scratch!

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## 🎯 What You'll Build

A production-ready image classification API that:
- ✅ Trains a CNN on CIFAR-10 dataset
- ✅ Tracks experiments with MLflow
- ✅ Serves predictions via REST API
- ✅ Runs in Docker containers

---

## 📚 Workshop Path

Follow these branches in order to build the complete pipeline:

### **Branch 1: `01-basic-training`** - Train Your First Model
Learn the basics of PyTorch training.

```bash
git checkout 01-basic-training
```

**What you'll do:**
- Load CIFAR-10 dataset
- Train a simple CNN
- Save the trained model

[📖 See Branch README](https://github.com/noyzzz/cifar10-mlops-pipeline/tree/01-basic-training)

---

### **Branch 2: `workshop-add-mlflow`** - Add Experiment Tracking
Track your experiments like a pro.

```bash
git checkout workshop-add-mlflow
```

**What you'll do:**
- Add MLflow tracking (7 easy TODOs)
- Log hyperparameters and metrics
- Compare multiple training runs
- View results in MLflow UI

[📖 See Branch README](https://github.com/noyzzz/cifar10-mlops-pipeline/tree/workshop-add-mlflow)

---

### **Branch 3: `main`** - Deploy with API & Docker
You're here! Now let's deploy your model.

**What you'll do:**
1. Train your model with MLflow tracking
2. Build a REST API to serve predictions
3. Containerize everything with Docker

---

## 🚀 Quick Start (Local Deployment)

### **Step 1: Train the Model**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (creates artifacts/best_model.pt)
python -m src.ml.train --epochs 10

# View experiments (optional)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

### **Step 2: Run the API Locally**

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Test health check
curl http://localhost:8000/health

# Test prediction (use your own image!)
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# View interactive docs
open http://localhost:8000/docs
```

**Expected Response:**
```json
{
  "filename": "test_image.jpg",
  "predictions": [
    {"label": "airplane", "prob": 0.85},
    {"label": "ship", "prob": 0.10},
    {"label": "automobile", "prob": 0.03}
  ]
}
```

---

### **Step 3: Deploy with Docker**

Now let's containerize it for production!

#### **Install Docker**
If you don't have Docker: https://www.docker.com/products/docker-desktop/

After installation, open the Docker Desktop application.

#### **Build & Run**

```bash
# Build the Docker image
docker compose build

# Start the container
docker compose up -d

# Check it's running
docker compose ps

# Test the containerized API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# View logs
docker compose logs -f

# Stop when done
docker compose down
```

**That's it!** 🎉 Your model is now running in a production-ready container!

---

## 📖 API Documentation

### **Endpoints**

#### Health Check
```bash
GET /health
```

#### Predict Image Class
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image.jpg>
```

#### Interactive Docs
Visit http://localhost:8000/docs for Swagger UI

---

## 🧪 Test with Sample Images

Download CIFAR-10 test images:
```bash
# The training script already downloads data to ./data/
# You can use any image from there, or use your own!

# Example: Use an airplane image
curl -X POST http://localhost:8000/predict \
  -F "file=@./my_airplane.jpg" | python -m json.tool
```

---

## 📊 Model Info

**Architecture:** SimpleCifarCNN
- 2 Convolutional layers
- 2 Max pooling layers  
- 2 Fully connected layers
- ~543K parameters

**Performance:**
- Validation Accuracy: 65-75%
- Inference Time: ~50ms (CPU)
- Model Size: 2.1 MB

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🛠️ Troubleshooting

### **Port already in use**
```bash
# Change port in docker-compose.yml or use different port:
uvicorn src.api.main:app --port 8001
```

### **Model not found**
```bash
# Make sure you trained first:
python -m src.ml.train --epochs 10
ls artifacts/  # Should show best_model.pt
```

### **MLflow UI not working**
```bash
# Kill existing process
pkill -9 python

# Start on different port
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 🎓 What You Learned

After completing this workshop, you can:
- ✅ Train deep learning models with PyTorch
- ✅ Track experiments with MLflow
- ✅ Build REST APIs with FastAPI
- ✅ Deploy models with Docker
- ✅ Create production-ready ML pipelines

---

## 📂 Project Structure

```
cifar10-mlops-workshop/
├── src/
│   ├── ml/
│   │   ├── model.py      # CNN architecture
│   │   ├── data.py       # Data loading
│   │   ├── train.py      # Training with MLflow
│   │   └── infer.py      # Inference logic
│   └── api/
│       └── main.py       # FastAPI app
├── artifacts/            # Trained models
├── tests/               # Unit tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Service orchestration
└── requirements.txt     # Dependencies
```

---

## 🚀 Next Steps

Want to learn more?

- **Add CI/CD**: Check `.github/workflows/ci.yml` for automated testing
- **Improve Model**: Try different architectures, data augmentation
- **Add Monitoring**: Log predictions, track model drift
- **Scale Up**: Deploy to cloud (AWS, GCP, Azure)

---

## 🤝 Contributing

Found a bug or have suggestions? Open an issue or PR!

---

## 📧 Contact

**Authors:**
- [@noyzzz](https://github.com/noyzzz) - Navid
- [@meghakalia](https://github.com/meghakalia) - Megha Kalia
- [@alejoaa](https://github.com/alejoaa) - Alejandro Aguirre

**Repository:** [github.com/noyzzz/cifar10-mlops-pipeline](https://github.com/noyzzz/cifar10-mlops-pipeline)

---

⭐ **Star this repo if you found it helpful!**
