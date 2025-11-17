# src/ml/data.py
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
    val_split: float = 0.1,  # 10% of training for validation
    seed: int = 42,  # For reproducible splits
) -> Tuple[DataLoader, DataLoader]:
    """
    Split CIFAR-10 training set into train/validation.
    Keep test set separate for final evaluation.
    
    Returns:
        train_loader: 45,000 images with augmentation
        val_loader: 5,000 images without augmentation
    """
    # Load training set
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=_train_transform())
    
    # Calculate split sizes
    train_size = int((1 - val_split) * len(full_train))  # 45,000
    val_size = len(full_train) - train_size              # 5,000
    
    # Create reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,  # No shuffle for validation
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def get_test_loader(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Get test set for final evaluation only."""
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=_test_transform())
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

CLASSES = ("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")
