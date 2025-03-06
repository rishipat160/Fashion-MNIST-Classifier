import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Feedforward Neural Network for Fashion-MNIST Classification

This file contains the model definition for a feedforward neural network
designed to classify Fashion-MNIST images. The network accepts flattened
28x28 grayscale images (784 input features) and outputs predictions across
10 clothing categories.
"""

class FF_Net(nn.Module):
    def __init__(self):
        """
        Initialize the feedforward network architecture with:
        - 5 fully connected layers with decreasing sizes
        - Batch normalization after each layer except the output
        - ReLU activation functions
        - Dropout for regularization
        """
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            Logits tensor of shape [batch_size, 10]
        """
        x = x.view(-1, 784)  # Flatten the input images
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
        


