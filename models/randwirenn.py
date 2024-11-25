import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_channels, output_size, cnn_layers, rnn_hidden_layers, wire_density=0.8, rnn_input_size=512):
        super(RandWiReNN, self).__init__()

        # CNN Layers
        self.cnn_layers = nn.ModuleList()
        in_channels = input_channels
        for _ in range(cnn_layers):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
            )
            in_channels = 32

        # Calculate flattened size for RNN input
        cnn_output_size = 32 * (224 // (2 ** cnn_layers)) * (224 // (2 ** cnn_layers))

        # RNN Layers
        self.rnn_layers = nn.ModuleList()
        prev_size = cnn_output_size
        for hidden_size in rnn_hidden_layers:
            layer = self.create_random_layer(prev_size, hidden_size, wire_density)
            self.rnn_layers.append(layer)
            prev_size = hidden_size

        # Output Layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def create_random_layer(self, in_features, out_features, wire_density):
        """Create a layer with random connections (weights)."""
        linear_layer = nn.Linear(in_features, out_features, bias=True)
        weight_mask = (torch.rand(out_features, in_features) < wire_density).float()
        with torch.no_grad():
            linear_layer.weight.data *= weight_mask
        return linear_layer

    def forward(self, x):
        # CNN Layers
        for layer in self.cnn_layers:
            x = layer(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # RNN Layers
        for layer in self.rnn_layers:
            x = F.relu(layer(x))

        # Output Layer
        x = self.output_layer(x)
        return x
