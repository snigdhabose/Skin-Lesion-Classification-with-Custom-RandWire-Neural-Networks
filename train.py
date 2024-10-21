import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyperparameters
input_size = 224 * 224 * 3
output_size = 7
hidden_layers = [1024, 512, 256]
learning_rate = 0.001
epochs = 10 # 50
batch_size = 64

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        
        # Map string labels to integers
        self.label_mapping = {
            'mel': 0,      # melanoma
            'nv': 1,       # nevus
            'bcc': 2,      # basal cell carcinoma
            'akiec': 3,    # actinic keratoses
            'bkl': 4,      # benign keratosis-like lesions
            'df': 5,       # dermatofibroma
            'vasc': 6      # vascular lesions
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 1]  # Assuming the filename is in column 1
        img_name += ".jpg"  # Add the file extension
        
        # Try to load from part 1, then part 2
        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found in part 1 or part 2 directories.")
        
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label_str = self.metadata.iloc[idx, 2]  # Assuming the label is in column 2 as string
        
        # Convert label from string to integer
        label = self.label_mapping[label_str]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split the dataset
full_dataset = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',
    img_dir_part2='./data/HAM10000_images_part_2/',
    transform=None
)

train_idx, val_idx = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=full_dataset.metadata.iloc[:, 2]
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model, criterion, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RandWiReNN(input_size, output_size, hidden_layers, wire_density=0.5, dropout_rate=0.5).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training and validation loop
def train_and_validate():
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)  # Flatten and send to device
            labels = labels.to(device)  # Send labels to device
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.size(0), -1).to(device)  # Flatten and send to device
                labels = labels.to(device)  # Send labels to device
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_randwirenn_model.pth')
            print("New best model saved.")

    print("Training completed. Best model saved as 'best_randwirenn_model.pth'")

if __name__ == "__main__":
    train_and_validate()
