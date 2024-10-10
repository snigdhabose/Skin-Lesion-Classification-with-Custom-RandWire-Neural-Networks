import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from models.randwirenn import RandWiReNN
import os

print("Running main.py")

# Hyperparameters
input_size = 28 * 28  # Example for MNIST
output_size = 10       # Example for MNIST (digits 0-9)
hidden_layers = [512, 256]  # Adjust as needed
batch_size = 64

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

# Load the MNIST dataset (replace with your own dataset)
# Load the MNIST dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
print("Number of training samples:", len(train_dataset))  # Check the size of the dataset

test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNN(input_size, output_size, hidden_layers, wire_density=0.5)
# Test the model with a dummy input
dummy_input = torch.randn(1, input_size)  # Create a dummy input tensor
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
        # print(f"Training with batch: inputs shape {inputs.shape}, labels shape {labels.shape}")
        optimizer.zero_grad()
        outputs = model(inputs.view(inputs.size(0), -1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(f"Batch loss: {loss.item()}")
    
    print(f'Epoch {epoch + 1}/10 completed, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'skin_lesion_model.pth')
print("Model saved.")
# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)

# Testing
model.eval()
with torch.no_grad():  # Disable gradient calculation for testing
    correct = 0
    total = 0
    for inputs, labels in test_loader:  # Testing data
        outputs = model(inputs.view(inputs.size(0), -1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2f}')
