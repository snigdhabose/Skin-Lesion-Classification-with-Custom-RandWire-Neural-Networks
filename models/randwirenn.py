import torch
import torch.nn as nn
import torch.nn.functional as F

class RandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5, dropout_rate=0.5):
        super(RandWiReNN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layer = self.create_random_layer(prev_size, hidden_size, wire_density)
            self.layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            print(f"Created layer: input size {prev_size}, output size {hidden_size}")

        self.output_layer = nn.Linear(prev_size, output_size)
        print(f"Created output layer: input size {prev_size}, output size {output_size}")

    def create_random_layer(self, in_features, out_features, wire_density):
        linear_layer = nn.Linear(in_features, out_features, bias=True)
        weight_mask = (torch.rand(out_features, in_features) < wire_density).float()
        
        with torch.no_grad():
            linear_layer.weight.data *= weight_mask
            print(f"Layer created with weight mask applied: in_features={in_features}, out_features={out_features}, wire_density={wire_density}")

        return linear_layer

    def forward(self, x):
        x = x.view(x.size(0), -1)

        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = F.relu(bn(layer(x)))
            x = dropout(x)

        x = self.output_layer(x)
        return x