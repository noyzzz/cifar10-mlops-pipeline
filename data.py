"""Data loading utilities for CIFAR-10."""
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def _test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = False,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=_train_transform()
    )
    
    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
