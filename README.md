# Fashion-MNIST Classifier

A deep learning project that implements and compares two neural network architectures for classifying clothing items from the Fashion-MNIST dataset.

## Overview

This project explores image classification using both Feedforward Neural Networks (FFN) and Convolutional Neural Networks (CNN) on the Fashion-MNIST dataset. The implementation is in PyTorch and includes:

- Model architecture design and implementation
- Training and evaluation pipelines
- Visualization of model performance
- Convolutional kernel visualization

The Fashion-MNIST dataset consists of 70,000 grayscale images of clothing items across 10 categories, serving as a more challenging drop-in replacement for the traditional MNIST dataset.

## Project Structure

- `fashionmnist.py`: Main script for data loading, model training, and evaluation
- `ffn.py`: Implementation of the Feedforward Neural Network
- `cnn.py`: Implementation of the Convolutional Neural Network
- `kernel_vis.py`: Script for visualizing CNN kernels and their transformations

## Models

### Feedforward Neural Network (FFN)
- 5 fully connected layers with decreasing sizes
- Batch normalization after each layer except the output
- ReLU activation functions
- Dropout for regularization

### Convolutional Neural Network (CNN)
- 3 convolutional layers with batch normalization
- Max pooling after each convolutional layer
- 3 fully connected layers
- Dropout for regularization

## Setup and Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Fashion-MNIST-Classifier.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Fashion-MNIST-Classifier
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script to train and evaluate the models:
    ```bash
    python fashionmnist.py
    ```

5. Visualize the CNN kernels:
    ```bash
    python kernel_vis.py
    ```

This will train and evaluate both the FFN and CNN models, and generate plots for the training loss.

## Results

The project generates several visualization outputs:

- Training loss curves for both models
- Example predictions (both correct and incorrect) for each model
- Visualization of CNN kernels from the first layer
- Visualization of how these kernels transform input images

The CNN model typically achieves higher accuracy than the FFN model, demonstrating the effectiveness of convolutional layers for image classification tasks.

## Sample Outputs

After running the scripts, you'll find the following output files:
- `ffn_loss.png` and `cnn_loss.png`: Training loss curves
- `ffn_correct.png`, `ffn_incorrect.png`, `cnn_correct.png`, `cnn_incorrect.png`: Example predictions
- `kernel_grid.png`: Visualization of CNN kernels
- `image_transform_grid.png`: Visualization of kernel transformations

## Disclaimer

This project is for educational purposes only and is not intended for production use. The models and visualizations are provided as examples of how to implement and visualize neural network architectures for image classification tasks.
This was done as a project for the course CS5100 at Northeastern University.
