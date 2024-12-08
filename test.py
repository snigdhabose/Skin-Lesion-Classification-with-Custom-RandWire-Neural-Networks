import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 224 * 224 * 3  # Flattened size of input image
output_size = 7
batch_size = 64
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Class labels

# Custom dataset for HAM10000
class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        self.label_mapping = {label: idx for idx, label in enumerate(class_names)}

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

        return image.view(-1), label, img_name  # Flatten image for RandWiReNN

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load test dataset
test_data = HAM10000Dataset(
    csv_file='./data/HAM10000_metadata.csv',
    img_dir_part1='./data/HAM10000_images_part_1/',
    img_dir_part2='./data/HAM10000_images_part_2/',
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNN(input_size=input_size, output_size=output_size, hidden_layers=[1024, 512, 256], wire_density=0.8)
model.load_state_dict(torch.load('best_randwirenn_model.pth'))
model.to(device)
model.eval()

# Evaluation on test set
correct = 0
total = 0
class_correct = [0] * output_size
class_total = [0] * output_size


correct_images = []
correct_predictions = []
correct_labels = []

with torch.no_grad():
    for images, labels, img_names in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_correct[label] += (pred == label)
            class_total[label] += 1

            # Collect only correctly classified samples
            if pred == label and len(correct_images) < 5:  
                correct_images.append(images[i].cpu())
                correct_predictions.append(pred)
                correct_labels.append(label)

# Print overall accuracy
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Print class-wise accuracy
for i in range(output_size):
    if class_total[i] > 0:
        print(f'Accuracy of {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    else:
        print(f'No samples for {class_names[i]}')

# Visualize correctly classified images
for i in range(len(correct_images)):
    img = correct_images[i].reshape(3, 224, 224).permute(1, 2, 0).numpy() * 0.5 + 0.5  
    plt.imshow(img)
    plt.title(f"True: {class_names[correct_labels[i]]}, Predicted: {class_names[correct_predictions[i]]}")
    plt.axis('off')
    plt.show()
