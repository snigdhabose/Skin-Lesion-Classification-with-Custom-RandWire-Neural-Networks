import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5):
        super(RandWiReNN, self).__init__()

        print("Initializing RandWiReNN...")  

        # Random-wired fully connected layers
        self.layers = nn.ModuleList()
        prev_size = input_size

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
        # Pass through hidden layers
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        
        # Pass through the output layer
        x = self.output_layer(x)
        
        return x
