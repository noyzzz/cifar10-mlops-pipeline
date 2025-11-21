"""Batch inference helper for CIFAR-10 raw batch files.

Reads a CIFAR batch (pickle format, e.g. `data/cifar-10-batches-py/test_batch`),
loads the model via `src.ml.infer.load_model`, and writes predictions to CSV
or prints them. Also optionally compares predictions to true labels when present.

Usage:
  python -m src.ml.infer_batch --batch data/cifar-10-batches-py/test_batch --out preds.csv
"""
from __future__ import annotations
import io
import os
import sys
import csv
import pickle
from typing import Optional

import numpy as np
from PIL import Image

from src.ml.infer import load_model, predict_image_bytes
from src.ml.data import CLASSES


def load_cifar_batch(path: str):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    return batch


def row_to_image_bytes(row) -> bytes:
    arr = np.asarray(row, dtype=np.uint8)
    arr = arr.reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="JPEG")
    return bio.getvalue()


def infer_batch(
    batch_path: str,
    model_path: Optional[str] = None,
    classes_path: Optional[str] = None,
    out_csv: Optional[str] = None,
    topk: int = 1,
    subset: Optional[int] = None,
):
    model, classes, device = load_model(
        checkpoint_path=model_path or "artifacts/best_model.pt",
        classes_path=classes_path or "artifacts/classes.txt",
    )

    batch = load_cifar_batch(batch_path)
    data = batch.get("data")
    labels = batch.get("labels") or batch.get("fine_labels")

    # Prepare CSV writer if requested
    writer = None
    out_file = None
    if out_csv:
        out_file = open(out_csv, "w", newline="")
        writer = csv.writer(out_file)
        header = ["index", "pred_label", "pred_prob"]
        if labels is not None:
            header.insert(1, "true_label")
        writer.writerow(header)

    # Respect subset if provided (None or <=0 means process all)
    total = len(data)
    if subset is None or subset <= 0:
        limit = total
    else:
        limit = min(total, subset)

    # Collect results so caller (e.g. API) can use them programmatically
    results = []
    for idx in range(limit):
        row = data[idx]
        img_bytes = row_to_image_bytes(row)
        res = predict_image_bytes(model, classes, img_bytes, device, topk=topk)
        topk_res = res.get("topk", [])

        # Build a compact prediction entry
        preds = [{"label": p["label"], "prob": float(p["prob"])} for p in topk_res]

        if labels is not None:
            true_idx = labels[idx]
            true_label = CLASSES[true_idx] if 0 <= true_idx < len(CLASSES) else str(true_idx)
        else:
            true_label = None

        entry = {
            "index": int(idx),
            "true_label": true_label,
            "predictions": preds,
        }

        results.append(entry)

        # If CSV writer requested, write a single-row summary (top1)
        if writer:
            top1_label = preds[0]["label"] if preds else ""
            top1_prob = preds[0]["prob"] if preds else 0.0
            row_out = [idx]
            if labels is not None:
                row_out.append(true_label)
            row_out.extend([top1_label, f"{top1_prob:.6f}"])
            writer.writerow(row_out)

    if out_file:
        out_file.close()


    return {"batch": os.path.basename(batch_path), "count": len(results), "results": results}

    if out_file:
        out_file.close()
    return {"topk": [{"pred label": v, "true label": i} for v, i in zip(pred_label, true_label)]}


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Infer on a CIFAR batch file and output predictions")
    p.add_argument("--batch", type=str, default="data/cifar-10-batches-py/test_batch", help="Path to CIFAR batch (pickle) file")
    p.add_argument("--model", type=str, default=None, help="Path to model checkpoint (state_dict)")
    p.add_argument("--classes", type=str, default=None, help="Path to classes.txt")
    p.add_argument("--out", type=str, default=None, help="CSV output path (if omitted, prints to stdout)")
    p.add_argument("--topk", type=int, default=1, help="Top-K predictions to return (affects prob shown)")
    p.add_argument("--subset", type=int, default = 100, help = "number of test files for prediction")
    
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    batch_path = args.batch
    if not os.path.exists(batch_path):
        print(f"Batch file not found: {batch_path}", file=sys.stderr)
        sys.exit(2)

    result = infer_batch(
        batch_path=batch_path,
        model_path=args.model,
        classes_path=args.classes,
        out_csv=args.out,
        topk=args.topk,
        subset=args.subset,
    )

    # print pred and true labels 
    print(result.get('results'))
    


if __name__ == "__main__":
    main()
