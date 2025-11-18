"""
Simple prediction script.
Load a model and predict on a single image.
"""
import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN
import sys

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(model_path='best_model.pt'):
    """Load trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image


def predict(model, image, device):
    """Make prediction on image."""
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return predicted.item(), confidence.item()


def main():
    """Main prediction function."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py cat.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load model
    print("Loading model...")
    model, device = load_model()
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = preprocess_image(image_path)
    
    # Predict
    print("Making prediction...")
    class_idx, confidence = predict(model, image, device)
    class_name = CLASSES[class_idx]
    
    # Display result
    print(f"\n{'='*40}")
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    main()
