"""Inference utilities for CIFAR-10 model."""
import io
from typing import Dict, Tuple, List

import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCifarCNN


_preproc = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


def _load_classes(path: str = "artifacts/classes.txt") -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_model(
    checkpoint_path: str = "artifacts/best_model.pt",
    classes_path: str = "artifacts/classes.txt",
    device: torch.device = None,
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
    return _preproc(img).unsqueeze(0)


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
    return {
        "topk": [
            {"label": classes[i.item()], "prob": float(v)} 
            for v, i in zip(vals, idxs)
        ]
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Predict CIFAR-10 class")
    p.add_argument("--image", required=True, help="Image path")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    print("Loading model...")
    model, classes, device = load_model()
    print(f"Loading image: {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    print("Making prediction...")
    result = predict_image_bytes(model, classes, image_bytes, device, topk=args.topk)
    
    print(f"\n{'='*40}")
    print(f"Top {args.topk} Predictions:")
    print(f"{'='*40}")
    for i, pred in enumerate(result["topk"], 1):
        print(f"{i}. {pred['label']:<12} {pred['prob']*100:>6.2f}%")
    print(f"{'='*40}\n")
