import torch
from PIL import Image
from torchvision import transforms
from model import CNNModel
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = CNNModel().to(device)
model.load_state_dict(torch.load("saved_model/mnist_cnn.pth"))
model.eval()

# Image path
img_path = 'images/sample_digit.png'

# Load and preprocess image
image = Image.open(img_path).convert('L')  # Convert to grayscale

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
])

input_tensor = transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).item()

# Show prediction
print(f"\n🔍 Predicted Digit: {prediction}")

# Optional: Show the image with prediction title
plt.imshow(image, cmap='gray')
plt.title(f'Predicted Digit: {prediction}', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("images/inferred_digit.png")
plt.show()
print("🖼️ Saved visualization to images/inferred_digit.png")
