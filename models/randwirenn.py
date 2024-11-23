# randwirenn_with_resnet_features.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class RandWiReNNWithResNetFeatures(nn.Module):
    def __init__(self, input_channels, output_size):
        super(RandWiReNNWithResNetFeatures, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottleneck blocks with skip connections
        self.bottleneck1 = BottleneckBlock(64, 32, 64)
        self.bottleneck2 = BottleneckBlock(64, 32, 64)

        # Downsampling layer
        self.downsample = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Additional bottleneck block after downsampling
        self.bottleneck3 = BottleneckBlock(128, 64, 128)

        # Placeholder for fully connected layer (init with dummy size)
        self.fc = nn.Linear(128 * 56 * 56, output_size)  # Initial size, will update

        # Initialize fully connected layer size dynamically
        self._initialize_fc()

    def forward(self, x):
        # Initial conv layer with batch normalization
        x = F.relu(self.bn1(self.initial_conv(x)))

        # First bottleneck block with residual connection
        residual = x
        x = self.bottleneck1(x) + residual  # Skip connection

        # Second bottleneck block with residual connection
        residual = x
        x = self.bottleneck2(x) + residual  # Skip connection

        # Downsampling with batch normalization
        x = F.relu(self.bn2(self.downsample(x)))

        # Third bottleneck block with residual connection
        residual = x
        x = self.bottleneck3(x) + residual  # Skip connection

        # Flatten and output
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_fc(self):
        # Pass a dummy input through the model to compute the correct flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = F.relu(self.bn1(self.initial_conv(dummy_input)))
            x = self.bottleneck1(x) + x
            x = self.bottleneck2(x) + x
            x = F.relu(self.bn2(self.downsample(x)))
            x = self.bottleneck3(x) + x
            flattened_size = x.view(1, -1).size(1)

        # Redefine the fully connected layer with the correct input size
        self.fc = nn.Linear(flattened_size, self.fc.out_features)
