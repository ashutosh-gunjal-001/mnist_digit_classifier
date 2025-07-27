import torch
from model import CNNModel
from data_loader import get_data_loaders
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CNNModel().to(device)
model.load_state_dict(torch.load("saved_model/mnist_cnn.pth"))
model.eval()

# Load test data
_, test_loader = get_data_loaders()
y_true, y_pred = [], []

# Inference
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
end_time = time.time()

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"🕒 Evaluation Time: {end_time - start_time:.2f} seconds")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=range(10), yticklabels=range(10))
plt.title("MNIST Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
print("🖼️ Saved confusion matrix to images/confusion_matrix.png")
