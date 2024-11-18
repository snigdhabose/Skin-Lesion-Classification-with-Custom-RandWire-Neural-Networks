import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from models.randwirenn import RandWiReNN

# Hyperparameters
input_channels = 3  # Number of input channels (RGB)
output_size = 7  # Number of classes in HAM10000
hidden_layers = [512, 256, 128]
learning_rate = 0.00001
epochs = 150
batch_size = 32

# Custom Dataset for HAM10000
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        
        self.label_mapping = {
            'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 'bkl': 4, 'df': 5, 'vasc': 6
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 1] + ".jpg"
        
        # Try to load image from part 1, then part 2
        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found in part 1 or part 2 directories.")
        
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label = torch.tensor(self.label_mapping[self.metadata.iloc[idx, 2]], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset and DataLoader
train_data = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',  
    img_dir_part2='./data/HAM10000_images_part_2/',  
    transform=transform
)

# Class counts and WeightedRandomSampler setup
class_counts = train_data.metadata['dx'].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[train_data.label_mapping[train_data.metadata.iloc[i]['dx']]] for i in range(len(train_data))]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)

# Initialize model
model = RandWiReNN(input_channels, output_size, hidden_layers, wire_density=0.5)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Save the model
    torch.save(model.state_dict(), 'randwirenn_model.pth')
    print("Model saved as 'randwirenn_model.pth'")

if __name__ == "__main__":
    train()