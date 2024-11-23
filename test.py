# test.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from models.randwirenn import RandWiReNNWithResNetFeatures
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
output_size = 7
batch_size = 32

# Custom dataset for HAM10000
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
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found in directories.")

        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.label_mapping[self.metadata.iloc[idx]['dx']], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load test dataset
dataset = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',
    img_dir_part2='./data/HAM10000_images_part_2/',
    transform=transform
)

# Assuming you have a separate test set, but for demonstration, we'll use the validation set
from sklearn.model_selection import train_test_split
_, test_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.metadata['dx'],
    random_state=42
)

test_subset = Subset(dataset, test_indices)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNNWithResNetFeatures(input_channels=3, output_size=output_size)
model.load_state_dict(torch.load('best_randwirenn_model.pth'))
model.to(device)
model.eval()

# Evaluation on test set
correct = 0
total = 0
class_correct = list(0. for i in range(output_size))
class_total = list(0. for i in range(output_size))

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Print overall accuracy
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Print class-wise accuracy
for i in range(output_size):
    if class_total[i] > 0:
        print(f'Accuracy of class {i} : {100 * class_correct[i] / class_total[i]:.2f}%')
    else:
        print(f'No samples for class {i}')
