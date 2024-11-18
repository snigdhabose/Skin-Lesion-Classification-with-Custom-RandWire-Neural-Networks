import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# Hyperparameters
input_size = 224  # Assuming the image input size is 224x224
output_size = 7   # Number of classes in HAM10000
batch_size = 32

# Custom Dataset for HAM10000 (reusing the dataset class as in train.py)
import os
import torch
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from models.randwirenn import RandWiReNN  # Import your RandWiReNN model

# Custom dataset for HAM10000 (reused from train.py)
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform

        # Map string labels to integers dynamically
        unique_labels = self.metadata.iloc[:, 2].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx, 1] + ".jpg"  # Add file extension

        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found in part 1 or part 2 directories.")

        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label_str = self.metadata.iloc[idx, 2]  # Assuming the label is in column 2
        label = self.label_mapping[label_str]  # Convert label to integer
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define your data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the HAM10000 dataset for testing
test_data = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',
    img_dir_part2='./data/HAM10000_images_part_2/',
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNN(input_size=224, output_size=output_size, hidden_layers=[512, 256, 128], wire_density=0.5)
model.load_state_dict(torch.load('randwirenn_model.pth'))
model.eval()

# Evaluation on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print accuracy
print(f'Test Accuracy: {100 * correct / total:.2f}%')