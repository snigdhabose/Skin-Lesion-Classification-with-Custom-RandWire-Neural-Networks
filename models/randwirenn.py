# randwirenn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RandWiReNN(nn.Module):
    def __init__(self, output_size, hidden_layers, wire_density=0.8):
        super(RandWiReNN, self).__init__()

        # Load a pretrained ResNet18 model
        self.cnn_model = models.resnet18(pretrained=True)

        # Freeze initial layers to retain pretrained features
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # Unfreeze the last convolutional block
        for param in self.cnn_model.layer4.parameters():
            param.requires_grad = True

        # Replace the fully connected layer with Identity to get features
        num_ftrs = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Identity()

        # Random-wired fully connected layers
        self.layers = nn.ModuleList()
        prev_size = num_ftrs
        self.dropout = nn.Dropout(p=0.5)  # Add dropout for regularization

        for hidden_size in hidden_layers:
            layer = self.create_random_layer(prev_size, hidden_size, wire_density)
            self.layers.append(layer)
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def create_random_layer(self, in_features, out_features, wire_density):
        """Create a layer with random connections (weights)."""
        linear_layer = nn.Linear(in_features, out_features, bias=True)
        weight_mask = (torch.rand(out_features, in_features) < wire_density).float()
        with torch.no_grad():
            linear_layer.weight.data *= weight_mask
        return linear_layer

    def forward(self, x):
        # Pass through pretrained CNN model
        x = self.cnn_model(x)

        # Pass through random-wired fully connected layers
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)  # Apply dropout

        # Output layer
        x = self.output_layer(x)
        return x
