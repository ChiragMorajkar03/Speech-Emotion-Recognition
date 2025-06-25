# Model Files

This directory contains the trained model files for the Speech Emotion Recognition project.

## Required Files

The following model files are required for the application to work properly:

1. `my_model.keras` - Main TensorFlow model file
2. `my_model.json` - Model architecture in JSON format
3. `my_model_weights.weights.h5` - Model weights file

## Obtaining the Models

These files are not included in the GitHub repository due to their large size. You can download them from the following link:

[Download Model Files](https://drive.google.com/drive/folders/your-folder-id) (Replace with your actual download link)

## Installation

After downloading the model files, place them in this directory (`models/`).

## Training Your Own Model

If you want to train your own model, refer to the Jupyter notebook `Speech Emotion Recognition - Sound Classification.ipynb` which contains the complete model training process.

## Model Architecture

The emotion recognition model uses a CNN architecture with the following layers:
- Multiple convolutional layers
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

The model was trained on standard emotion recognition datasets and achieves approximately 75% accuracy on the test set.
