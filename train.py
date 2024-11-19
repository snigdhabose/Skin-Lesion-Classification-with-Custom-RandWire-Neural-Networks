import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.randwirenn import RandWiReNN

# Check for Metal (MPS) availability
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because PyTorch was not built with MPS enabled")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ and/or you don't have an MPS-enabled device")
    device = torch.device("cpu")
else:
    device = torch.device("mps")

print(f"Using device: {device}")

# Hyperparameters
input_size = (3, 224, 224)  # 3-channel (RGB), 224x224 images
output_size = 7  # Number of classes in HAM10000
hidden_layers = [512, 256, 128]
learning_rate = 0.001
epochs = 20
batch_size = 32

# Custom Dataset for HAM10000
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        
        # Map string labels to integers
        self.label_mapping = {
            'mel': 0,
            'nv': 1,
            'bcc': 2,
            'akiec': 3,
            'bkl': 4,
            'df': 5,
            'vasc': 6
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 1] + ".jpg"
        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found.")
        
        image = Image.open(img_path).convert('RGB')
        label_str = self.metadata.iloc[idx, 2]
        label = self.label_mapping[label_str]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


# Define data augmentation and preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the HAM10000 dataset
train_data = HAM10000Dataset(csv_file='./data/HAM10000_metadata.csv',
                             img_dir_part1='./data/HAM10000_images_part_1/',  
                             img_dir_part2='./data/HAM10000_images_part_2/',  
                             transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Update the model architecture to include convolutional layers
class EnhancedRandWiReNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, wire_density=0.5):
        super(EnhancedRandWiReNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        conv_output_size = 64 * (input_size[1] // 4) * (input_size[2] // 4)
        self.randwirenn = RandWiReNN(conv_output_size, output_size, hidden_layers, wire_density)

    def forward(self, x):
        x = self.conv_layers(x)  # Apply convolutional layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.randwirenn(x)  # Pass through RandWiReNN
        return x

# Initialize the model
model = EnhancedRandWiReNN(input_size, output_size, hidden_layers, wire_density=0.5)
model = model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

# Training loop
def train():
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()  # Adjust learning rate
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    torch.save({'model_state_dict': model.cpu().state_dict()}, 'enhanced_randwirenn_model.pth')
    print("Model saved as 'enhanced_randwirenn_model.pth'")

if __name__ == "__main__":
    train()
