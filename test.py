import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class HAM10000Dataset(Dataset):
    # ... (keep the existing implementation)
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

def test(model, test_loader, device):
    model.eval()
    print("Starting testing phase...")

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.view(inputs.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = sum(p == t for p, t in zip(all_predictions, all_labels)) / len(all_labels)
    print(f'\nTest Accuracy: {accuracy:.4f}')

    # Generate and print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=test_dataset.label_mapping.keys()))

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.label_mapping.keys(), 
                yticklabels=test_dataset.label_mapping.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the HAM10000 test dataset
    test_dataset = HAM10000Dataset(csv_file='./data/HAM10000_metadata.csv',
                                   img_dir_part1='./data/HAM10000_images_part_1/',
                                   img_dir_part2='./data/HAM10000_images_part_2/',
                                   transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load your RandWiReNN model with the correct input/output size for HAM10000
    model = RandWiReNN(input_size=224 * 224 * 3, output_size=7, hidden_layers=[1024, 512, 256], dropout_rate=0.5)
    model.load_state_dict(torch.load('best_randwirenn_model.pth'))
    model = model.to(device)
    
    # Run the test
    test(model, test_loader, device)