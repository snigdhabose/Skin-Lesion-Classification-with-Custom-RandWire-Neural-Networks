import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.randwirenn import RandWiReNN

# Hyperparameters
input_size = 224 * 224 * 3  # Example input size for RGB images resized to 224x224
output_size = 7  # Number of classes in HAM10000
hidden_layers = [512, 256, 128]
learning_rate = 0.001
epochs = 10
batch_size = 32

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


# Define your data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load the HAM10000 dataset, including both part 1 and part 2
train_data = HAM10000Dataset(csv_file='./data/HAM10000_metadata.csv',
                             img_dir_part1='./data/HAM10000_images_part_1/',  
                             img_dir_part2='./data/HAM10000_images_part_2/',  
                             transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the model
model = RandWiReNN(input_size, output_size, hidden_layers, wire_density=0.5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train():
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)  # Flatten the images

            optimizer.zero_grad()
            output = model(images)
            
            # Compute loss
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    # Save the model after training is completed
    torch.save(model.state_dict(), 'randwirenn_model.pth')
    print("Model saved as 'randwirenn_model.pth'")

if __name__ == "__main__":
    train()
