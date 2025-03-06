import cv2
import numpy
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
import matplotlib.pyplot as plt

"""
CNN Kernel Visualization

This script visualizes the convolutional kernels from the first layer of the trained CNN model
and demonstrates how these kernels transform input images. It helps to understand what features
the CNN is detecting in the early layers of the network.
"""

# Load the trained CNN model
conv_net = Conv_Net()
conv_net.load_state_dict(torch.load('cnn.pth'))

# Extract the weights of the first convolutional layer
weights = conv_net.conv1.weight.data
num_kernels = weights.shape[0]

# Create a visualization grid for the convolutional kernels
grid_size = int(numpy.ceil(numpy.sqrt(num_kernels)))
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(num_kernels):
    kernel = weights[i, 0].detach().numpy()
    # Normalize kernel values to [0,1] for visualization
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')

# Save the kernel visualization
plt.suptitle('First Layer Kernels', fontsize=15, y=0.95)
plt.savefig('kernel_grid.png', bbox_inches='tight', dpi=300)
plt.close()

# Load a sample image and prepare it for convolution
img = cv2.imread('cnn_incorrect.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0  # Normalize the image
img = torch.tensor(img).float()
img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

print(f"Image shape: {img.shape}")

# Apply each kernel to the image
output = torch.nn.functional.conv2d(img, weights, padding=1)

# Reshape for visualization
output = output.squeeze(0)
output = output.unsqueeze(1)

# Create a visualization grid for the transformed images
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(num_kernels):
    transformed = output[i, 0].detach().numpy()
    # Normalize for better visualization
    transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min())
    
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(transformed, cmap='gray')
    plt.axis('off')

# Save the transformed image visualization
plt.suptitle('Image After Each Kernel', fontsize=16, y=0.95)
plt.savefig('image_transform_grid.png', bbox_inches='tight', dpi=300)
plt.close()














