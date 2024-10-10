import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinLesionModel(nn.Module):
    def __init__(self):
        super(SkinLesionModel, self).__init__()
        
        # Define your layers
        self.fc1 = nn.Linear(28 * 28, 512)  # Fully connected layer: input 28*28=784, output 512
        self.fc2 = nn.Linear(512, 10)        # Fully connected layer: input 512, output 10 (number of classes)
        print("SkinLesionModel initialized with layers.")

    def forward(self, x):
        print(f"Input shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the input tensor to (batch_size, 28*28)
        print(f"Input shape after flattening: {x.shape}")
        x = F.relu(self.fc1(x))     # Apply ReLU activation to the output of fc1
        print(f"Output shape after fc1: {x.shape}")
        x = self.fc2(x)              # Pass the output to the second layer
        print(f"Output shape after fc2: {x.shape}")
        return x  # Return the final output

# Example of creating an instance of the model
model = SkinLesionModel()
