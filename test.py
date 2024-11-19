import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN  # Import your RandWiReNN model
import pandas as pd
from PIL import Image
import os

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

# Test function
def test(model):
    # Ensure the model is in evaluation mode
    model.eval()
    print("Starting testing phase...")

    # Define transformations (adjust these according to your dataset)
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # Resize the image to match the input size used during training
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
    # ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the HAM10000 test dataset
    test_dataset = HAM10000Dataset(csv_file='./data/HAM10000_metadata.csv',
                                   img_dir_part1='./data/HAM10000_images_part_1/',
                                   img_dir_part2='./data/HAM10000_images_part_2/',
                                   transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for testing
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs.view(inputs.size(0), -1))  # Flatten the images
            _, predicted = torch.max(outputs.data, 1)

            # Debugging information for each batch
            print(f"\nBatch {i+1}:")
            print(f"True labels:    {labels.tolist()}")
            print(f"Predicted labels: {predicted.tolist()}")
            
            # Track total correct predictions
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'\nTest Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    # Load your RandWiReNN model with the correct input/output size for HAM10000
    model = RandWiReNN(input_size=224 * 224 * 3, output_size=7, hidden_layers=[512, 256, 128])  # Adjust as necessary
    
    # Load the weights from the correctly trained model (HAM10000)
    model.load_state_dict(torch.load('randwirenn_model.pth'))
    
    # Run the test
    test(model)
