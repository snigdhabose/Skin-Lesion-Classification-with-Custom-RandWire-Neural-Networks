import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5):
        super(RandWiReNN, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        
        # Create random wired layers
        for hidden_size in hidden_layers:
            layer = self.create_random_layer(prev_size, hidden_size, wire_density)
            self.layers.append(layer)
            prev_size = hidden_size
            print(f"Created layer: input size {prev_size}, output size {hidden_size}")

        # Final output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        print(f"Created output layer: input size {prev_size}, output size {output_size}")

    def create_random_layer(self, in_features, out_features, wire_density):
        """Create a layer with random connections (weights)."""
        linear_layer = nn.Linear(in_features, out_features, bias=True)
        
        # Create the weight mask to match the shape of the weight matrix
        weight_mask = (torch.rand(out_features, in_features) < wire_density).float()
        
        # Adjust the weights based on the mask
        with torch.no_grad():
            linear_layer.weight.data *= weight_mask
            print(f"Layer created with weight mask applied: in_features={in_features}, out_features={out_features}, wire_density={wire_density}")

        bn = nn.BatchNorm1d(out_features)
        dropout = nn.Dropout(p=0.3)  # You can adjust the dropout rate

        # return linear_layer
        return nn.Sequential(linear_layer, bn, dropout)

    def forward(self, x):
        # print(f"Input shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the input
        # print(f"Input shape after flattening: {x.shape}")

        # Pass through all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))
            # print(f"Output shape after layer: {x.shape}")

        # Pass through the output layer
        x = self.output_layer(x)
        # print(f"Output shape after final layer: {x.shape}")
        return x

