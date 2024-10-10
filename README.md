# Skin-Lesion-Classification-with-Custom-RandWire-Neural-Networks
Implementing a custom RandWire Neural Network for efficient skin lesion classification using the HAM10000 dataset.

# RandWiReNN Skin Lesion Classifier

This project trains a neural network (RandWiReNN) for skin lesion classification using the HAM10000 dataset.

## Prerequisites
1. Ensure you have Python 3.6 or later installed.
2. Clone this repository.
3. Install the necessary dependencies using `requirements.txt`.

## Dataset Setup
You need to download the HAM10000 dataset and place it in the `data/` directory within the project.

### Steps:
1. Download the dataset from Kaggle:
   - Go to [HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
   - Download and extract the dataset.
   - Rename the folder to `data`.
   - Place the `data` folder in the root of the project directory.

The directory structure should look like this:

randwirenn_project/  
│  
├── data/                 # Dataset directory (HAM10000)  
├── models/               # Model definitions  
│   └── randwirenn.py     # RandWiReNN model definition  
├── utils/                # Helper functions and utilities  
│   └── __init__.py  
├── main.py               # Main script for running the project  
├── train.py              # Script for training the model  
├── test.py               # Script for testing the model  
└── requirements.txt      # List of dependencies  


## Install dependencies:
Make sure you have the necessary libraries installed. Install them by running:
pip install -r requirements.txt

## Running the Project

1. Running the main.py file

The main.py file initializes and runs the RandWiReNN model.

Run the following command:

python main.py

You should see output like:
Running main.py
Number of training samples: 60000
Layer created with weight mask applied: in_features=784, out_features=512, wire_density=0.5
...
Epoch 10/10 completed, Loss: 0.023572664707899094
Model saved.
Test Accuracy: 0.98


This will generate the trained model and save it as randwirenn_model.pth.

2. Running the train.py file
The train.py script is used to train the model on the HAM10000 dataset. Run it using the following command:
python train.py

Example output:
Layer created with weight mask applied: in_features=150528, out_features=512, wire_density=0.5
...
Epoch 10/10, Loss: 0.6863396700959616
Model saved as 'randwirenn_model.pth'

3. Running the test.py file
Once the model has been trained and saved, you can evaluate it by running the test.py script.

Run the following command:
python test.py


Conclusion

After running all the steps above, you will have a trained model (randwirenn_model.pth) that can classify skin lesions.
