# src/ml/train.py
import argparse
import os
import random
from typing import Dict

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

from src.ml.data import get_dataloaders, CLASSES
from src.ml.model import SimpleCifarCNN

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # ok for demo
    torch.backends.cudnn.benchmark = False

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, device, criterion, optimizer) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, count = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, targets) * bs
        count += bs

    return {"loss": total_loss / count, "acc": total_acc / count}

def evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, targets) * bs
            count += bs

    return {"loss": total_loss / count, "acc": total_acc / count}

def save_best(model, out_dir="artifacts", filename="best_model.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(model.state_dict(), path)
    return path

def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny CIFAR-10 CNN with MLflow logging")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", type=str, default="cifar10_cnn")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleCifarCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Local file-based MLflow tracking inside the repo
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)

    best_val_acc = 0.0
    best_path = None

    with mlflow.start_run():
        # Log static params
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

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_metrics = evaluate(model, val_loader, device, criterion)

            # Per-epoch metrics
            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
            }, step=epoch)

            print(f"[{epoch:02d}/{args.epochs}] "
                  f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_path = save_best(model)

        # Log the best checkpoint and class labels as artifacts
        if best_path:
            mlflow.log_artifact(best_path, artifact_path="artifacts")
        # Optionally log as an MLflow model for easy loading later
        example = torch.randn(1, 3, 32, 32).numpy()
        mlflow.pytorch.log_model(model, artifact_path="model", input_example=example)

        # Keep the best metric handy
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Log classes (useful for inference)
        with open("artifacts/classes.txt", "w") as f:
            f.write("\n".join(CLASSES))
        mlflow.log_artifact("artifacts/classes.txt", artifact_path="artifacts")

    print(f"Training done. Best val_acc={best_val_acc:.4f}. "
          f"Artifacts in mlruns/ and ./artifacts/")

if __name__ == "__main__":
    main()
