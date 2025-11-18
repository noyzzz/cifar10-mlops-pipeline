# 🚀 CIFAR-10 MLOps Pipeline

Complete end-to-end MLOps pipeline for CIFAR-10 image classification using PyTorch, MLflow, FastAPI, and Docker.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## 📋 Overview

This project demonstrates a complete MLOps workflow:
1. **Train** a compact CNN on CIFAR-10 dataset
2. **Track** experiments with MLflow
3. **Serve** the model via FastAPI REST API
4. **Containerize** with Docker for production deployment
5. **Monitor** model performance and predictions

## 🏗️ Architecture

```
cifar10-mlops-pipeline/
├── src/
│   ├── ml/
│   │   ├── model.py      # SimpleCifarCNN architecture
│   │   ├── data.py       # Data loaders with train/val split
│   │   ├── train.py      # Training pipeline with MLflow
│   │   └── infer.py      # Inference logic
│   └── api/
│       └── main.py       # FastAPI application
├── artifacts/            # Trained model artifacts
│   ├── best_model.pt     # Best model weights
│   └── classes.txt       # Class labels
├── tests/               # Unit tests
├── Dockerfile           # Production API container
├── docker-compose.yml   # Service orchestration
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### Option 1: Docker (Recommended for Production)

#### Install Docker if you haven't 

If you don't already have Docker installed, download it here:

https://www.docker.com/products/docker-desktop/

After installation, open the Docker Desktop application.

#### Build and Start the API

```bash
# Build and start the API
docker compose build
docker compose up -d

# Test the API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"

# View API documentation
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.ml.train --epochs 20 --batch-size 128

# Start MLflow UI (optional)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start API server
uvicorn src.api.main:app --reload

# Test the API
curl http://localhost:8000/health
```

## 📊 Training

### Basic Training
```bash
python -m src.ml.train --epochs 20 --batch-size 128
```

### Advanced Training with Custom Parameters
```bash
python -m src.ml.train \
  --experiment "cifar10_experiment" \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.0001
```

### Training Parameters
- `--experiment`: MLflow experiment name (default: "cifar10_test")
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--weight-decay`: L2 regularization (default: 0.0001)

### Expected Performance
- **Training time**: ~5-10 minutes (10 epochs, CPU)
- **Validation accuracy**: 65-75% (simple CNN)
- **Model size**: ~2.1 MB

## 🔍 MLflow Tracking

View experiment results and model artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Logged metrics:**
- Training loss/accuracy per epoch
- Validation loss/accuracy per epoch
- Best validation accuracy
- Model parameters (lr, batch_size, etc.)

## 🌐 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Image Class
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image>
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cat.jpg" \
  | python -m json.tool
```

**Response:**
```json
{
  "filename": "cat.jpg",
  "predictions": [
    {"label": "cat", "prob": 0.7234},
    {"label": "dog", "prob": 0.1523},
    {"label": "deer", "prob": 0.0456}
  ]
}
```

### Interactive Documentation
Visit `http://localhost:8000/docs` for Swagger UI with interactive API testing.

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the image
docker compose build

# Start services
docker compose up -d

# Check logs
docker compose logs -f

# Stop services
docker compose down
```

### Environment Variables
- `MODEL_PATH`: Path to model file (default: `/app/artifacts/best_model.pt`)
- `CLASSES_PATH`: Path to classes file (default: `/app/artifacts/classes.txt`)

## 📦 Model Architecture

**SimpleCifarCNN:**
- 2 Convolutional layers (32, 64 filters)
- 2 Max pooling layers
- 2 Fully connected layers
- ReLU activation
- ~543K parameters

**Input:** 32×32 RGB images  
**Output:** 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test specific file
pytest tests/test_infer.py

# With coverage
pytest --cov=src tests/
```

## 📈 Performance Benchmarks

| Metric | Value |
|--------|-------|
| Validation Accuracy | 65-75% |
| Inference Time (CPU) | ~50ms |
| Model Size | 2.1 MB |
| Docker Image Size | ~1.2 GB |
| API Response Time | ~100ms |

## 🛠️ Development

### Project Structure
```
src/
├── ml/
│   ├── model.py    # Neural network architecture
│   ├── data.py     # Dataset loading & preprocessing
│   ├── train.py    # Training loop & MLflow logging
│   └── infer.py    # Model loading & prediction
└── api/
    └── main.py     # FastAPI application
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- [PyTorch](https://pytorch.org/) for deep learning framework
- [MLflow](https://mlflow.org/) for experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) for API framework

## 📧 Contact

**Author:** Navid  
**Repository:** [github.com/noyzzz/cifar10-mlops-pipeline](https://github.com/noyzzz/cifar10-mlops-pipeline)

---

⭐ If you find this project helpful, please give it a star!
