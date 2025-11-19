# 🎓 CIFAR-10 Training with MLflow Workshop

**Learn to add experiment tracking to your ML code!**

This workshop teaches you how to add MLflow to track experiments, compare runs, and save models.

## 📁 What You Have

```
├── data.py          # Data loading with train/val split
├── model.py         # CNN model definition
├── train.py         # Training script (with TODOs for you!)
├── infer.py         # Inference script
└── requirements.txt # Includes mlflow
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Training (Without MLflow)

```bash
# Train for 5 epochs - this works without MLflow!
python train.py --epochs 5
```

This saves your model to `artifacts/best_model.pt` but doesn't track anything.

**Problem:** You can't compare different runs or see how metrics changed over time.

---

## 🎯 Your Task: Add MLflow Tracking

Open `train.py` and uncomment 7 TODO sections. Here's what each one does:

### **TODO 1: Import MLflow** (Lines 6-7)
```python
import mlflow
import mlflow.pytorch
```
**Why:** Brings in MLflow's tracking capabilities.

---

### **TODO 2: Add Experiment Argument** (Line 83)
```python
p.add_argument("--experiment", type=str, default="cifar10_cnn")
```
**Why:** Lets you name your experiments (e.g., "low_lr", "high_batch").

---

### **TODO 3: Set Up Tracking** (Lines 101-102)
```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(args.experiment)
```
**Why:** Creates a local database to store all your experiments.

---

### **TODO 4: Start Run** (Line 110)
```python
with mlflow.start_run():
```
**⚠️ Important:** Indent everything below this line (lines 111-155) by 4 spaces!

**Why:** Everything inside this block gets tracked automatically.

---

### **TODO 5: Log Hyperparameters** (Lines 113-122)
```python
mlflow.log_params({
    "model": "SimpleCifarCNN",
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "num_workers": args.num_workers,
    "device": str(device),
    "seed": args.seed,
})
```
**Why:** Records settings so you can see what worked best later.

---

### **TODO 6: Log Metrics** (Lines 128-133)
```python
mlflow.log_metrics({
    "train_loss": train_metrics["loss"],
    "train_acc": train_metrics["acc"],
    "val_loss": val_metrics["loss"],
    "val_acc": val_metrics["acc"],
}, step=epoch)
```
**Why:** Tracks loss/accuracy after each epoch so you can plot learning curves.

---

### **TODO 7: Log Artifacts** (Lines 149-158)
```python
if best_path:
    mlflow.log_artifact(best_path, artifact_path="artifacts")
mlflow.log_artifact("artifacts/classes.txt", artifact_path="artifacts")

example = torch.randn(1, 3, 32, 32).numpy()
mlflow.pytorch.log_model(model, artifact_path="model", input_example=example)

mlflow.log_metric("best_val_acc", best_val_acc)
```
**Why:** Saves your trained model and files so you can reload them later.

---

## ✅ Test Your Changes

### 1. Run Training with MLflow

```bash
python train.py --epochs 5
```

You should see the same output as before, but now everything is tracked!

### 2. View Results in MLflow UI

Open a **new terminal**:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser: **http://localhost:5000**

You'll see:
- 📊 **Metrics** plotted over epochs (train/val loss and accuracy)
- 📋 **Parameters** for this run (lr, batch_size, etc.)
- 📦 **Artifacts** (your trained model files)

### 3. Compare Multiple Runs

Try different settings:

```bash
# Experiment 1: Low learning rate
python train.py --epochs 5 --lr 0.0001

# Experiment 2: High learning rate
python train.py --epochs 5 --lr 0.01

# Experiment 3: Large batch size
python train.py --epochs 5 --batch-size 256
```

In MLflow UI:
- Select multiple runs
- Click **"Compare"** 
- See which hyperparameters worked best! 🎉

---

## 🎓 What You Learned

✅ How to add MLflow to existing training code  
✅ How to log hyperparameters, metrics, and models  
✅ How to use MLflow UI to analyze experiments  
✅ How to compare multiple training runs  

---

## ➡️ Next Steps

Want to see the full MLOps pipeline? Check out the `main` branch:

```bash
git checkout main
```

It includes:
- ✅ MLflow tracking (what you just learned!)
- ✅ FastAPI REST API for serving predictions
- ✅ Docker for deployment
- ✅ GitHub Actions for CI/CD

---

## 💡 Troubleshooting

**Port 5000 already in use?**
```bash
# Use a different port
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

**IndentationError?**
Make sure you indented everything after `with mlflow.start_run():` (line 110)

**Want to start fresh?**
```bash
rm -rf mlflow.db mlruns/
python train.py --epochs 5
```
