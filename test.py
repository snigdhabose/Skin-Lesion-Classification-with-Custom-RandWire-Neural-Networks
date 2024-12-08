import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
output_size = 7
batch_size = 64

# Custom dataset for HAM10000
class HAM10000Dataset:
    def __init__(self, csv_file, img_dir_part1, img_dir_part2, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir_part1 = img_dir_part1
        self.img_dir_part2 = img_dir_part2
        self.transform = transform
        self.label_mapping = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
        }
        self.label_names = {v: k for k, v in self.label_mapping.items()}

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

        return image, label, img_name


# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Match ResNet's normalization
                         std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_data = HAM10000Dataset(
    csv_file='./Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks/data/HAM10000_metadata.csv',
    img_dir_part1='./Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks/data/HAM10000_images_part_1/',
    img_dir_part2='./Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks/data/HAM10000_images_part_2/',
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RandWiReNN(output_size=output_size, hidden_layers=[512, 256, 128], wire_density=0.8)
model.load_state_dict(torch.load('best_randwirenn_model.pth'))
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0
class_correct = [0] * output_size
class_total = [0] * output_size

# For visualization
sample_images = []
sample_predictions = []
sample_labels = []
sample_names = []

print("Starting evaluation...")
with torch.no_grad():
    for batch_idx, (images, labels, img_names) in enumerate(test_loader):
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

            
            if len(sample_images) < 5 and pred == label:
                sample_images.append(images[i].cpu().permute(1, 2, 0))  
                sample_predictions.append(pred)
                sample_labels.append(label)
                sample_names.append(img_names[i])

# Print overall accuracy
print(f"\nTest Accuracy: {100 * correct / total:.2f}%\n")

# Print class-wise accuracy
label_mapping = test_data.label_mapping
label_names = {v: k for k, v in label_mapping.items()}
for i in range(output_size):
    if class_total[i] > 0:
        print(f"Accuracy of {label_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
    else:
        print(f"No samples for {label_names[i]}")

# Visualize correctly classified images
print("\nVisualizing correctly classified images:")
for i, image in enumerate(sample_images):
    plt.figure()
    plt.imshow((image.numpy() * 0.229 + 0.485).clip(0, 1))  
    plt.title(
        f"Prediction: {label_names[sample_predictions[i]]} | True Label: {label_names[sample_labels[i]]}"
    )
    plt.axis("off")
    plt.savefig(f"visualization_correct_model3_img_{sample_names[i]}.png")
    plt.show()
