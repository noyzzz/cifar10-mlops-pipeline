# 🎓 Basic CIFAR-10 Training

**Simple PyTorch training - No MLOps!**

This is the starting point before we add MLOps practices.

## 📁 Files

```
├── data.py        # Data loading
├── model.py       # CNN model
├── train.py       # Training script
├── infer.py       # Inference script
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py --epochs 5
```

This will:
- Download CIFAR-10 dataset  
- Train for 5 epochs with train/val split
- Save `artifacts/best_model.pt` and `artifacts/classes.txt`

### 3. Predict

```bash
python infer.py --image cat.jpg --topk 3
```

Output:
```
Top 3 Predictions:
1. cat          78.45%
2. dog          12.34%
3. deer          5.67%
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
