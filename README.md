# Skin Lesion Classification with Custom RandWiRe Neural Networks

This project explores the use of RandWire Neural Networks (RandNNs) for skin lesion classification using the HAM10000 dataset. The repository contains three custom architectures:

1. **Model 1**: RandWiReNN from scratch.
2. **Model 2**: RandWiReNN combined with CNN.
3. **Model 3**: RandWiReNN with a pretrained ResNet-18 backbone.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset-structure)
- [Project Structure](#project-structure)
- [Branches and Model Details](#branches-and-model-details)
- [Configurations](#configurations)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)

---

## Introduction

Skin lesion classification is a challenging task due to the complexity of medical images and the imbalanced distribution of classes. RandNNs, with their randomly wired layers, offer a promising alternative for extracting robust features and addressing these challenges.

This repository implements three architectures to evaluate the performance of RandNNs in conjunction with other strategies like CNNs and pretrained backbones.

---

## Dataset Structure

The project uses the **HAM10000** dataset, which contains 10,015 dermatoscopic images across seven classes. 

**Classes:**
- `akiec`: Actinic keratoses and intraepithelial carcinoma.
- `bcc`: Basal cell carcinoma.
- `bkl`: Benign keratosis-like lesions.
- `df`: Dermatofibroma.
- `mel`: Melanoma.
- `nv`: Melanocytic nevi.
- `vasc`: Vascular lesions.

The **HAM10000** dataset should be organized under the `data/` directory in the root of the repository. 

### Files Description
- **HAM10000_metadata.csv**: Contains metadata about the images, including class labels (`dx`) and image IDs.
- **HAM10000_images_part_1/** and **HAM10000_images_part_2/**: Folders containing dermatoscopic images.

### How to Add the Dataset
1. Download the HAM10000 dataset from Kaggle: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Extract the files and place them in the `data/` directory as per the structure above.
3. Ensure that all image files are correctly organized into `HAM10000_images_part_1/` and `HAM10000_images_part_2/`, with the corresponding metadata in `HAM10000_metadata.csv`.

### Note
The dataset is critical for running the training and testing processes. Ensure the folder structure and file locations are accurate to avoid runtime errors.

---

## Project Structure

- **data/**: Contains the dataset required for training and testing. You need to download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it in this directory.
  - `HAM10000_metadata.csv`: Metadata file for the dataset.
  - `HAM10000_images_part_1/` and `HAM10000_images_part_2/`: Subdirectories containing dermatoscopic images.

- **models/**: Includes model definitions for the project.
  - `randwirenn.py`: Defines the RandWiReNN architecture for all three models.

- **utils/**: Contains helper scripts for reusable functionality.
  - `__init__.py`: Placeholder for utilities (add more helper scripts as needed).

- **main.py**: Entry point for running the project. Handles initialization and coordination between modules.

- **train.py**: Script for training the models, including data preprocessing, augmentation, and early stopping.

- **test.py**: Script for testing the trained models and evaluating their performance.

- **requirements.txt**: Contains the Python dependencies required for the project. Install them using `pip install -r requirements.txt`.

---

## Branches and Model Details

Each model is implemented in a separate branch. Switch to the corresponding branch to explore the model.

### Model 1: RandWiReNN
- **Branch:** `rnn_from_scratch_sb_6874`
- **Architecture:** RandWiReNN built from scratch with fully connected layers.
- **File to check:** `models/randwirenn.py`
- **Training script:** `train.py`
- **Testing script:** `test.py`

### Model 2: RandWiReNN + CNN
- **Branch:** `rnn_with_cnn_sb_7511`
- **Architecture:** Combines CNN layers for feature extraction with RandWiReNN for classification.
- **File to check:** `models/randwirenn.py`
- **Training script:** `train.py`
- **Testing script:** `test.py`

### Model 3: RandWiReNN + ResNet
- **Branch:** `rnn_with_resnet_sb_9009`
- **Architecture:** Leverages a pretrained ResNet-18 backbone with RandWiReNN for final classification.
- **File to check:** `models/randwirenn.py`
- **Training script:** `train.py`
- **Testing script:** `test.py`

---

## Configurations

### Common Hyperparameters
- **Learning rate:** 0.0001
- **Weight decay:** 1e-5
- **Batch size:** 64
- **Epochs:** 150
- **Patience:** Early stopping patience set to 5–10 epochs.

### Data Augmentation
- Random horizontal and vertical flips.
- Random rotations.
- Resized cropping and color jittering.
- Resizing to 224x224 and normalization.

---

## Usage

### Setup Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/snigdhabose/Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks.git

2. Navigate to the project directory and install dependencies:
   ```bash
   cd Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks
   pip install -r requirements.txt

3. Run Training

Switch to the desired branch and train the corresponding model:
   ```bash
      git checkout <branch_name>
      python train.py
```

The train.py script is used to train the model on the HAM10000 dataset. Run it using the following command:
python train.py

Example output:
Layer created with weight mask applied: in_features=150528, out_features=512, wire_density=0.5
...
Epoch 10/10, Loss: 0.6863396700959616
Model saved as 'randwirenn_model.pth'


4. Run Testing

Evaluate the model on the test set:
```bash
python test.py
```

---
# Results

### Model Performance Summary
| **Model**                   | **Accuracy** | **Key Features**                           |
|-----------------------------|--------------|--------------------------------------------|
| **RandWiReNN (Model 1)**    | 68.05%       | Random wiring for fully connected layers.  |
| **RandWiReNN + CNN (Model 2)** | 75.11%    | CNN layers for feature extraction.         |
| **RandWiReNN + ResNet (Model 3)** | 90.09% | Pretrained ResNet-18 backbone.             |

### Observations
1. Model 1 provided a lightweight and fast baseline with limited spatial feature learning.
2. Model 2 added CNN-based feature extraction, leading to improved performance.
3. Model 3 leveraged a pretrained ResNet-18 backbone, achieving the highest accuracy by combining robust feature extraction and random wiring.

---

## Demo

Watch the demo showcasing all three models in action:  
[YouTube Demo Video](https://www.youtube.com/watch?v=Xbj9GYCKSHg)

---

## Architecture Diagrams

Architecture diagrams for all three models will be added to this section to provide a clear visual understanding of their designs and layer compositions.

- **RandWiReNN (Model 1)**: Randomly wired fully connected layers.  
  This model uses a unique approach where the connections between neurons are determined randomly, allowing it to capture unconventional patterns in the data. While lightweight and efficient, it lacks explicit spatial feature extraction, making it less suitable for complex image data.  
  <img src="https://github.com/user-attachments/assets/abf60f3c-d807-47bd-957a-fb589a1c0e72" alt="Model 1" width="300"/>

- **RandWiReNN + CNN (Model 2)**: Random wiring with CNN-based feature extraction.  
  By integrating CNN layers for hierarchical spatial feature extraction, this model improves its ability to process image data. The random wiring in the fully connected layers adds flexibility and generalization capability, striking a balance between lightweight design and performance.  
  <img src="https://github.com/user-attachments/assets/9cb4006d-48a0-4f94-9e39-791c64e26593" alt="Model 2" width="300"/>

- **RandWiReNN + ResNet (Model 3)**: Pretrained ResNet-18 for robust feature extraction integrated with random wiring.  
  This model leverages the powerful feature extraction capabilities of ResNet-18, pretrained on ImageNet. It fine-tunes the last convolutional block to adapt to the skin lesion dataset and uses random wiring in its classification layers for added generalization, achieving the best performance among the three.  
  <img src="https://github.com/user-attachments/assets/cce25201-cbb4-4f29-954b-46c575b6c882" alt="Model 3" width="300"/>



### Conclusion

This project demonstrates the effectiveness of **RandWiReNN** architectures for skin lesion classification. By progressing from simple random wiring to incorporating CNN and pretrained ResNet-18 backbones, we achieved significant accuracy improvements. The results highlight the importance of combining robust feature extraction techniques with adaptable architectures, especially for complex medical datasets like HAM10000.

### Future Work

- **Enhancing Generalization**: Explore additional regularization techniques such as label smoothing or mixup for better performance on unseen data.
- **Transfer Learning**: Experiment with other pretrained models like EfficientNet to further boost classification accuracy.
- **Deployment**: Integrate the best-performing model into a web or mobile application for real-time skin lesion analysis.

### Contributions

Contributions to this project are welcome! Feel free to submit issues or pull requests on the [GitHub repository](https://github.com/snigdhabose/Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks.git).

