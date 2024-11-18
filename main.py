import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.randwirenn import RandWiReNN

print("Running main.py")

# Hyperparameters
input_channels = 3  # 3 channels for RGB images
output_size = 10    # Example for MNIST, update if necessary
hidden_layers = [512, 256]
batch_size = 64

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are 224x224 for CNN input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

# Load the MNIST dataset (or replace with your dataset)
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
print("Number of training samples:", len(train_dataset))  # Check the size of the dataset

test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNN(input_channels, output_size, hidden_layers, wire_density=0.5)

# Test the model with a dummy input (simulating a batch of 1 RGB image of size 224x224)
dummy_input = torch.randn(1, 3, 224, 224)  # Corrected dummy input shape
dummy_output = model(dummy_input)  # Run it through the model
print("Dummy output shape:", dummy_output.shape)  # Print the output shape

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

# Training loop
for epoch in range(10):  # Change to a higher value as needed
    model.train()
    print(f"Starting Epoch {epoch + 1}/10")
    for inputs, labels in train_loader:  # Training data
        # Prepare inputs for CNN
        inputs = inputs.expand(-1, 3, 224, 224)  # Expand grayscale to 3 channels if using MNIST

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}/10 completed, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'skin_lesion_model.pth')
print("Model saved.")
print("Current Working Directory:", os.getcwd())

# Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.expand(-1, 3, 224, 224)  # Expand grayscale to 3 channels if using MNIST
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2f}')
