import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_channels, output_size, hidden_layers, wire_density=0.5):
        super(RandWiReNN, self).__init__()
        
        # Initial CNN layer
        self.cnn_layer = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the flattened size after the CNN layer
        # Assuming input image size of 224x224 and 2x downsampling from pooling
        cnn_output_size = 32 * (224 // 2) * (224 // 2)
        
        # Initialize layers
        self.layers = nn.ModuleList()
        prev_size = cnn_output_size
        
        # Create random-wired layers
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

        return linear_layer

    def forward(self, x):
        # Pass through initial CNN layer
        x = self.pool(F.relu(self.cnn_layer(x)))  # Shape after CNN layer
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Pass through all hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Pass through the output layer
        x = self.output_layer(x)
        return x