"""
Simple training script for CIFAR-10.
No MLflow, no experiment tracking, just basic training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def get_data_loaders(batch_size=128):
    """Create train and test data loaders."""
    
    # Simple transforms - just convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {running_loss / (batch_idx + 1):.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def main():
    """Main training loop."""
    # Configuration
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loaders
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Model, loss, optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'\nStarting training for {EPOCHS} epochs...\n')
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f'Epoch [{epoch + 1}/{EPOCHS}]')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')
            # Save classes list
            with open('classes.txt', 'w') as f:
                f.write('\n'.join(CLASSES))
            print(f'  ✅ Saved best model (accuracy: {best_acc:.2f}%)')
        
        print()
    
    print(f'Training completed! Best test accuracy: {best_acc:.2f}%')
    print(f'Model saved to: best_model.pt')
    print(f'Classes saved to: classes.txt')


if __name__ == '__main__':
    main()
