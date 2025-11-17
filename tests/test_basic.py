"""Basic tests to ensure core functionality works."""
import torch
from src.ml.model import SimpleCifarCNN


def test_model_can_be_created():
    """Test that model can be instantiated."""
    model = SimpleCifarCNN(num_classes=10)
    assert model is not None


def test_model_forward_pass():
    """Test model can process a batch of images."""
    model = SimpleCifarCNN(num_classes=10)
    model.eval()
    
    # Create dummy input (batch of 2 images)
    x = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"


def test_classes_file_exists():
    """Test that classes.txt exists and has 10 classes."""
    import os
    classes_path = "artifacts/classes.txt"
    
    assert os.path.exists(classes_path), f"{classes_path} not found"
    
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    assert len(classes) == 10, f"Expected 10 classes, got {len(classes)}"
