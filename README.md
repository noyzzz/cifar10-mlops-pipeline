# 🎓 Basic CIFAR-10 Training

**Simple PyTorch training - No MLOps!**

This is the starting point before we add MLOps practices.

## 📁 Files

```
├── model.py       # CNN model
├── train.py       # Training script
├── predict.py     # Prediction script
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
```

This will:
- Download CIFAR-10 dataset
- Train for 20 epochs (~10 minutes on CPU)
- Save `best_model.pt` and `classes.txt`

### 3. Predict

```bash
python predict.py cat.jpg
```

Output:
```
Prediction: cat
Confidence: 78.45%
```

## ❌ What's Wrong with This?

1. **No experiment tracking** - Can't compare runs
2. **No versioning** - Overwrites model every time
3. **No deployment** - Just a .pt file, no API
4. **Not reproducible** - Works on my machine only
5. **Everything manual** - No automation

## ➡️ See the Solution

```bash
git checkout main
```

The `main` branch shows how to fix all these problems with:
- ✅ MLflow for tracking
- ✅ FastAPI for serving
- ✅ Docker for deployment
- ✅ CI/CD for automation
