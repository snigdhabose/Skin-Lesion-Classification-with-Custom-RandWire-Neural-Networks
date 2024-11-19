import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
import pandas as pd
from PIL import Image
import os

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

# Hyperparameters (matching train.py)
input_size = (3, 224, 224)
output_size = 7
hidden_layers = [512, 256, 128]

# Custom Dataset for HAM10000
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
        img_name = self.metadata.iloc[idx, 1] + ".jpg"
        
        img_path = os.path.join(self.img_dir_part1, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir_part2, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_name} not found in either directory.")

        image = Image.open(img_path).convert('RGB')
        label_str = self.metadata.iloc[idx, 2]
        label = self.label_mapping[label_str]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Enhanced RandWiReNN architecture (matching train.py)
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
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.randwirenn(x)
        return x

def test(model):
    model = model.to(device)
    model.eval()
    print("Starting testing phase...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = HAM10000Dataset(
        csv_file='./data/HAM10000_metadata.csv',
        img_dir_part1='./data/HAM10000_images_part_1/',
        img_dir_part2='./data/HAM10000_images_part_2/',
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_correct = 0
    total_samples = 0
    class_correct = [0] * 7
    class_total = [0] * 7

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)  # No need to reshape inputs
            _, predicted = torch.max(outputs.data, 1)
            
            # Move to CPU for printing
            labels_cpu = labels.cpu()
            predicted_cpu = predicted.cpu()

            print(f"\nBatch {i+1}:")
            print(f"True labels:    {labels_cpu.tolist()}")
            print(f"Predicted labels: {predicted_cpu.tolist()}")
            
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            for label, prediction in zip(labels_cpu, predicted_cpu):
                class_correct[label] += (label == prediction).item()
                class_total[label] += 1

    accuracy = total_correct / total_samples
    print(f'\nOverall Test Accuracy: {accuracy:.2f}')

    class_names = ['Melanoma', 'Nevus', 'Basal Cell Carcinoma', 
                  'Actinic Keratoses', 'Benign Keratosis', 
                  'Dermatofibroma', 'Vascular Lesions']
    
    print("\nPer-class Accuracy:")
    for i in range(7):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'{class_names[i]}: {class_acc:.2f} ({class_correct[i]}/{class_total[i]})')

if __name__ == "__main__":
    # Initialize the enhanced model
    model = EnhancedRandWiReNN(input_size, output_size, hidden_layers, wire_density=0.5)
    
    # Load the checkpoint
    checkpoint = torch.load('enhanced_randwirenn_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    
    # Run the test
    test(model)