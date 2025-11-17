"""Inference helpers for CIFAR-10 model."""
# src/ml/infer.py
from __future__ import annotations
import io
from typing import Dict, Tuple, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from src.ml.model import SimpleCifarCNN

_preproc = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 size
    transforms.ToTensor(),        # Just convert to tensor, no normalization
])

def _load_classes(path: str = "artifacts/classes.txt") -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def load_model(
    checkpoint_path: str = "artifacts/best_model.pt",
    classes_path: str = "artifacts/classes.txt",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, List[str], torch.device]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = _load_classes(classes_path)
    model = SimpleCifarCNN(num_classes=len(classes))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, device

def _preprocess_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _preproc(img).unsqueeze(0)  # [1, 3, 32, 32]

@torch.no_grad()
def predict_image_bytes(
    model: torch.nn.Module,
    classes: List[str],
    image_bytes: bytes,
    device: torch.device,
    topk: int = 3,
) -> Dict:
    x = _preprocess_bytes(image_bytes).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    vals, idxs = torch.topk(probs, k=topk)
    return {"topk": [{"label": classes[i.item()], "prob": float(v)} for v, i in zip(vals, idxs)]}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    with open(args.image, "rb") as f:
        b = f.read()
    m, c, d = load_model()
    print(predict_image_bytes(m, c, b, d, topk=args.topk))
