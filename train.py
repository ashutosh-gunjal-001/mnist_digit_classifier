import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from model import CNNModel
from data_loader import get_data_loaders

def plot_training_loss(losses, filename="images/train_loss.png"):
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label='Training Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
train_loader, _ = get_data_loaders()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create directories
os.makedirs("saved_model", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Training loop
train_loss = []
start_time = time.time()

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    train_loss.append(avg_loss)

end_time = time.time()
print(f"Training completed in {(end_time - start_time):.2f} seconds")

# Save model
torch.save(model.state_dict(), "saved_model/mnist_cnn.pth")

# Plot training loss
plot_training_loss(train_loss)
