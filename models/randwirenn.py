import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5):
        super(RandWiReNN, self).__init__()

        print("Initializing RandWiReNN...")  # Debugging statement

        # Random-wired fully connected layers
        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_layers:
            # print(f"Creating random layer: {prev_size} -> {hidden_size} with wire density {wire_density}")  # Debugging
            layer = self.create_random_layer(prev_size, hidden_size, wire_density)
            self.layers.append(layer)
            prev_size = hidden_size

        # Output layer
        # print(f"Creating output layer: {prev_size} -> {output_size}")  # Debugging
        self.output_layer = nn.Linear(prev_size, output_size)

    def create_random_layer(self, in_features, out_features, wire_density):
        """Create a layer with random connections (weights)."""
        linear_layer = nn.Linear(in_features, out_features, bias=True)
        weight_mask = (torch.rand(out_features, in_features) < wire_density).float()
        with torch.no_grad():
            linear_layer.weight.data *= weight_mask
        return linear_layer

    def forward(self, x):
        # print(f"Input to forward: {x.shape}")  # Debugging
        # Pass through all random-wired layers
        for i, layer in enumerate(self.layers):
            # print(f"Layer {i}: Before activation {x.shape}")  # Debugging
            x = F.relu(layer(x))
            # print(f"Layer {i}: After activation {x.shape}")  # Debugging
        # Pass through the output layer
        x = self.output_layer(x)
        # print(f"Output shape: {x.shape}")  # Debugging
        return x
