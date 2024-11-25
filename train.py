import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from models.randwirenn import RandWiReNN
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_channels = 3
output_size = 7
cnn_layers = 3
rnn_hidden_layers = [512, 256, 128]
wire_density = 0.8
learning_rate = 0.0001
epochs = 150
batch_size = 64
patience = 10  # Allow longer patience for CNN-RNN

# Dataset
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        self.label_mapping = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx]['image_id'] + ".jpg"
        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.label_mapping[self.metadata.iloc[idx]['dx']], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
dataset = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',
    img_dir_part2='./data/HAM10000_images_part_2/',
    transform=transform
)

# Train-Validation Split
train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.metadata['dx'],
    random_state=42
)

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

# Class Weights for Imbalance
train_labels = [dataset[i][1] for i in train_indices]
class_sample_count = torch.tensor([(train_labels.count(t)) for t in torch.unique(torch.tensor(train_labels), sorted=True)])
class_weights = 1. / class_sample_count.float()
samples_weight = torch.tensor([class_weights[t] for t in train_labels])

sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Model Initialization
model = RandWiReNN(
    input_channels=input_channels,
    output_size=output_size,
    cnn_layers=cnn_layers,
    rnn_hidden_layers=rnn_hidden_layers,
    wire_density=wire_density
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
writer = SummaryWriter()

# Training Loop
def train():
    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.update(1)
        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_randwirenn_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping triggered!')
                break

    print("Training completed.")

if __name__ == "__main__":
    train()
