#!/usr/bin/env python3
"""Extract the first image from CIFAR-10 `test_batch` and save as `test_image.jpg`.

Usage:
  python3 scripts/extract_test_image.py [batch_path] [out_path]

Defaults:
  batch_path: data/cifar-10-batches-py/test_batch
  out_path: test_image.jpg
"""
import os
import sys
import pickle
from PIL import Image
import numpy as np


def load_cifar_batch(path):
    with open(path, 'rb') as f:
        # CIFAR pickles were created with Python 2; use latin1 encoding for keys
        batch = pickle.load(f, encoding='latin1')
    return batch


def extract_image_from_row(row):
    # row is length 3072: first 1024 R, next 1024 G, next 1024 B
    arr = np.asarray(row, dtype=np.uint8)
    arr = arr.reshape(3, 32, 32)
    # transpose to H, W, C
    img = arr.transpose(1, 2, 0)
    return img


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_batch = os.path.join(repo_root, 'data', 'cifar-10-batches-py', 'test_batch')
    default_out = os.path.join(repo_root, 'test_image.jpg')

    batch_path = sys.argv[1] if len(sys.argv) > 1 else default_batch
    out_path = sys.argv[2] if len(sys.argv) > 2 else default_out

    if not os.path.exists(batch_path):
        print(f"Error: batch file not found: {batch_path}")
        sys.exit(2)

    batch = load_cifar_batch(batch_path)
    data = batch.get('data')
    if data is None:
        print('Error: no "data" key found in batch file')
        sys.exit(3)

    # pick the first image
    first = data[0]
    img_arr = extract_image_from_row(first)

    img = Image.fromarray(img_arr)
    img.save(out_path, format='JPEG')
    print(f"Saved image to: {out_path}")


if __name__ == '__main__':
    main()
