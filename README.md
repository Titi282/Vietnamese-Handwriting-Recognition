# Vietnamese-Handwriting-Recognition


## Overview
This project develops a system for recognizing Vietnamese handwritten text using a **Convolutional Recurrent Neural Network (CRNN)** combined with **Connectionist Temporal Classification (CTC) Loss**. The system processes a dataset of handwritten images, builds a vocabulary of 187 characters (including Latin letters and Vietnamese accented characters), and achieves a **Char Error Rate (CER)** of approximately 0.15.

## Features
- Preprocesses handwritten images to a standardized size (32x128 pixels).
- Constructs a vocabulary of 187 characters for Vietnamese text recognition.
- Implements a CRNN model with convolutional layers, bidirectional LSTMs, and CTC Loss.
- Evaluates performance using Char Error Rate (CER) metric.

## Technologies
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep learning framework for model building and training.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **OpenCV**: Image processing and loading.
- **Matplotlib & Seaborn**: Visualization of training results and metrics.

## Dataset
The model is trained on a dataset of Vietnamese handwritten images. The data is split into:

- **Training**: ~82,432 samples
- **Validation**: ~10,368 samples
- **Testing**: ~10,368 samples

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Titi282/Vietnamese-Handwriting-Recognition.git
## Model Architecture
- **Input**: Grayscale images (32x128x1).
- **Convolutional Layers**: Extract features using Conv2D and MaxPooling2D.
- **Recurrent Layers**: Bidirectional LSTMs for sequence modeling.
- **Output**: Dense layer with softmax activation for character prediction.
- **Loss**: CTC Loss for aligning predicted sequences with ground truth.

## Results
- **Char Error Rate (CER)**: ~0.15 on the test set.
- Training includes callbacks like `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` for optimization.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- OpenCV
- Matplotlib
- Seaborn

## Acknowledgments
- Inspired by handwriting recognition research and Kaggle community contributions.
- Dataset sourced from [Google Drive Handwriting Images Dataset](https://drive.google.com/file/d/15ZECOsDc8bITa5nDLwZo8f4pczSfAEg8/view?usp=sharing).
