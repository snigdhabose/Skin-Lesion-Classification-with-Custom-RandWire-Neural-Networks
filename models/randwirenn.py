import torch
import torch.nn as nn
import torch.nn.functional as F


class RandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5):
        super(RandWiReNN, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)  # Add dropout
        prev_size = input_size

        # Create random wired layers
        for hidden_size in hidden_layers:
            layer = nn.Linear(prev_size, hidden_size, bias=True)
            self.layers.append(layer)
            self.mask_weights(layer, wire_density)
            prev_size = hidden_size

        # Final output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def mask_weights(self, layer, wire_density):
        with torch.no_grad():
            weight_mask = (torch.rand_like(layer.weight) < wire_density).float()
            layer.weight.data *= weight_mask
            nn.init.xavier_uniform_(layer.weight)  # Use Xavier initialization

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input

        # Pass through all hidden layers
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.01)
            x = self.dropout(x)  # Apply dropout

        # Pass through the output layer
        x = self.output_layer(x)
        return x
