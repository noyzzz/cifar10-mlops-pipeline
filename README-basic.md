# 🎓 Workshop: Basic CIFAR-10 Training

**Simple PyTorch training - No MLOps yet!**

This branch demonstrates a basic ML workflow **without** MLOps practices.

## 📁 What's Here?

```
├── model.py       # Simple CNN definition
├── train.py       # Basic training script
├── predict.py     # CLI prediction tool
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch torchvision pillow
```

### 2. Train the model

```bash
python train.py
```

**Output:**
```
Using device: cpu
Loading CIFAR-10 dataset...
Train samples: 50000
Test samples: 10000

Starting training for 20 epochs...

Epoch [1/20]
  Batch [100/391] Loss: 1.8234 Acc: 32.45%
  ...
  Train Loss: 1.6543 | Train Acc: 40.23%
  Test Loss:  1.4321 | Test Acc:  48.56%
  ✅ Saved best model (accuracy: 48.56%)

...

Training completed! Best test accuracy: 72.34%
Model saved to: best_model.pt
Classes saved to: classes.txt
```

### 3. Make predictions

```bash
python predict.py cat.jpg
```

**Output:**
```
Loading model...
Loading image: cat.jpg
Making prediction...

========================================
Prediction: cat
Confidence: 78.45%
========================================
```

## ❌ Problems with This Approach

This basic workflow has **serious limitations**:

### 1. 📉 **No Experiment Tracking**
- ❌ Can't compare different runs
- ❌ Don't know which hyperparameters worked best
- ❌ No way to reproduce exact results
- ❌ Metrics are lost after terminal closes

### 2. 🔄 **No Model Versioning**
- ❌ Overwrites `best_model.pt` every time
- ❌ Can't rollback to previous versions
- ❌ Lost track of model history
- ❌ No way to compare models

### 3. 🚀 **No Deployment Strategy**
- ❌ Just a `.pt` file - how to serve it?
- ❌ No API for applications to use
- ❌ Can't integrate with production systems
- ❌ Manual predictions only

### 4. 🔬 **No Reproducibility**
- ❌ Different results on different machines
- ❌ No containerization
- ❌ Dependency conflicts
- ❌ "Works on my machine" syndrome

### 5. 🤦 **Everything is Manual**
- ❌ Manual testing
- ❌ Manual deployment
- ❌ No automation
- ❌ No CI/CD

### 6. 🐛 **No Quality Assurance**
- ❌ No automated tests
- ❌ No validation pipeline
- ❌ Easy to introduce bugs
- ❌ No code quality checks

## ➡️ Next Steps

See how MLOps solves these problems:

1. **MLflow** for experiment tracking
2. **FastAPI** for model serving
3. **Docker** for containerization
4. **CI/CD** for automation

```bash
# Switch to the complete MLOps version
git checkout main
```

## 📚 What You'll Learn in Main Branch

- ✅ Experiment tracking with MLflow
- ✅ Model versioning and registry
- ✅ REST API with FastAPI
- ✅ Docker containerization
- ✅ Automated testing
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Production-ready deployment

---

**This branch is intentionally simple to show why MLOps is necessary!**
