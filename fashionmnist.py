import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn import *
from ffn import *
import matplotlib.pyplot as plt

"""
Fashion-MNIST Classification with PyTorch

This script implements and compares two neural network architectures for classifying
Fashion-MNIST images:
1. A feedforward neural network (FFN)
2. A convolutional neural network (CNN)

The script handles data loading, model training, evaluation, and visualization of results.
"""

# Data preprocessing configuration
transform = transforms.Compose([                            
    transforms.ToTensor(),                                  
    transforms.Normalize(mean=[0.5], std=[0.5])             
])

batch_size = 32

if __name__ == '__main__':
    # Load and prepare the Fashion-MNIST dataset
    trainset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,     
        download=True,  
        transform=transform  
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    testset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize models
    feedforward_net = FF_Net()
    conv_net = Conv_Net()

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_ffn = optim.Adam(feedforward_net.parameters(), lr=0.001)
    optimizer_cnn = optim.Adam(conv_net.parameters(), lr=0.003)

    # Training configuration
    num_epochs_ffn = 15
    num_epochs_cnn = 12

    # Learning rate schedulers to improve training
    scheduler_ffn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ffn, mode='min', factor=0.1, patience=3)
    scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cnn, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        min_lr=1e-6
    )

    # Train the feedforward network
    print("Training Feedforward Network...")
    ffn_losses = []

    for epoch in range(num_epochs_ffn):
        running_loss_ffn = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer_ffn.zero_grad()

            outputs = feedforward_net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer_ffn.step()
            
            running_loss_ffn += loss.item()
        
        epoch_loss = running_loss_ffn / len(trainloader)
        ffn_losses.append(epoch_loss)
        print(f'FFN Epoch {epoch + 1}, Loss: {epoch_loss}')
        scheduler_ffn.step(epoch_loss)

    print('Finished Training FFN')
    # Save the model
    torch.save(feedforward_net.state_dict(), 'ffn.pth')

    # Train the convolutional network
    print("Training Convolutional Network...")
    cnn_losses = []

    for epoch in range(num_epochs_cnn):
        running_loss_cnn = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer_cnn.zero_grad()

            outputs = conv_net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer_cnn.step()
            
            running_loss_cnn += loss.item()
        
        epoch_loss = running_loss_cnn / len(trainloader)
        cnn_losses.append(epoch_loss)
        print(f'CNN Epoch {epoch + 1}, Loss: {epoch_loss}')
        scheduler_cnn.step(epoch_loss)

    print('Finished Training CNN')
    torch.save(conv_net.state_dict(), 'cnn.pth')

    # Class labels for visualization
    class_labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    # Function to save prediction examples
    def save_prediction_image(image, true_label, pred_label, model_name, is_correct):
        plt.figure(figsize=(5,5))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'True: {class_labels[true_label]}\nPredicted: {class_labels[pred_label]}')
        plt.axis('off')
        status = "correct" if is_correct else "incorrect"
        plt.savefig(f'{model_name}_{status}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # Uncomment to load saved models instead of training
    # feedforward_net.load_state_dict(torch.load('ffn.pth'))
    # conv_net.load_state_dict(torch.load('cnn.pth'))

    # Model evaluation
    print("Evaluating models on test data...")
    correct_ffn = 0
    total_ffn = 0
    correct_cnn = 0
    total_cnn = 0

    feedforward_net.eval()
    conv_net.eval()

    with torch.no_grad(): 
        ffn_correct_found = False
        ffn_incorrect_found = False
        cnn_correct_found = False
        cnn_incorrect_found = False
        
        for data in testloader:
            images, labels = data
            
            # Evaluate FFN
            outputs_ffn = feedforward_net(images)
            _, predicted_ffn = torch.max(outputs_ffn.data, 1)
            total_ffn += labels.size(0)
            correct_ffn += (predicted_ffn == labels).sum().item()
            
            # Evaluate CNN
            outputs_cnn = conv_net(images)
            _, predicted_cnn = torch.max(outputs_cnn.data, 1)
            total_cnn += labels.size(0)
            correct_cnn += (predicted_cnn == labels).sum().item()
            
            # Save example predictions for visualization
            for i in range(len(labels)):
                # FFN examples
                if not ffn_correct_found and predicted_ffn[i] == labels[i]:
                    save_prediction_image(images[i], labels[i].item(), predicted_ffn[i].item(), 'ffn', True)
                    ffn_correct_found = True
                elif not ffn_incorrect_found and predicted_ffn[i] != labels[i]:
                    save_prediction_image(images[i], labels[i].item(), predicted_ffn[i].item(), 'ffn', False)
                    ffn_incorrect_found = True
                
                # CNN examples
                if not cnn_correct_found and predicted_cnn[i] == labels[i]:
                    save_prediction_image(images[i], labels[i].item(), predicted_cnn[i].item(), 'cnn', True)
                    cnn_correct_found = True
                elif not cnn_incorrect_found and predicted_cnn[i] != labels[i]:
                    save_prediction_image(images[i], labels[i].item(), predicted_cnn[i].item(), 'cnn', False)
                    cnn_incorrect_found = True

    # Print final accuracy results
    print(f'Accuracy of FFN on test images: {100 * correct_ffn / total_ffn:.2f}%')
    print(f'Accuracy of CNN on test images: {100 * correct_cnn / total_cnn:.2f}%')

    # Visualize training loss
    print("Generating visualization plots...")
    
    # Plot FFN Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs_ffn + 1), ffn_losses, marker='o')
    plt.title('FFN Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('ffn_loss.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot CNN Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs_cnn + 1), cnn_losses, marker='o')
    plt.title('CNN Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('cnn_loss.png', bbox_inches='tight', dpi=300)
    plt.close()