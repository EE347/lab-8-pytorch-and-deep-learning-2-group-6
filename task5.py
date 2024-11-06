import time
import cv2

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    nll_loss = torch.nn.NLLLoss()

    # Saving parameters
    best_train_loss_ce = 1e9
    best_train_loss_nll = 1e9

    # Loss lists
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []

    # Epoch Loop
    for epoch in range(1, 3):

        # Start timer
        t = time.time_ns()

        # Train the model
        model.train()
        train_loss_ce = 0
        train_loss_nll = 0

        # Batch Loop
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            # labels = labels.reshape(-1, 1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss_ce = cross_entropy_loss(outputs, labels)

            log_probs = F.log_softmax(outputs, dim=1)
            loss_nll = nll_loss(log_probs, labels)

            # Backward pass for CE Loss
            loss_ce.backward()
            optimizer.step()

            # Accumulate the loss
            train_loss_ce += loss_ce.item()
            train_loss_nll += loss_nll.item()

        # Test the model
        model.eval()
        test_loss_ce = 0
        test_loss_nll = 0
        correct = 0
        total = 0

        # Batch Loop
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the CE Loss
            loss_ce = cross_entropy_loss(outputs, labels)
            test_loss_ce += loss_ce.item()

            log_probs = F.log_softmax(outputs, dim=1)
            loss_nll = nll_loss(log_probs, labels)
            test_loss_nll += loss_nll.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

        # Print the epoch statistics
        print(f'Epoch: {epoch}, Train Loss (CE): {train_loss_ce / len(trainloader):.4f},'
              f'Test Loss: (CE) {test_loss_ce / len(testloader):.4f},'
              f'Train Loss: (NLL) {test_loss_nll / len(testloader):.4f},'
              f'Test Loss: (NLL) {test_loss_nll / len(testloader):.4f},'
              f'Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Update loss lists
        train_losses_ce.append(train_loss_ce / len(trainloader))
        test_losses_ce.append(test_loss_ce / len(testloader))
        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))

        # Update the best model (CE)
        if train_loss_ce < best_train_loss_ce:
            best_train_loss_ce = train_loss_ce
            torch.save(model.state_dict(), 'best_model_ce.pth')

        # Update the best model (NLL)
        if train_loss_nll < best_train_loss_nll:
            best_train_loss_nll = train_loss_nll
            torch.save(model.state_dict(), 'best_model_nll.pth')

        # Save the model
        torch.save(model.state_dict(), 'current_model.pth')

        # Create the loss plot
        plt.plot(train_losses_ce, label='Train Loss (CE)')
        plt.plot(test_losses_ce, label='Test Loss (CE)')
        plt.plot(train_losses_nll, label='Train Loss (NLL)')
        plt.plot(test_losses_nll, label='Test Loss (NLL)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('task5_loss_plot.png')

        print("\nEpoch-wise Loss Comparison Table")
        print("Epoch | Train Loss (CE) | Test Loss (CE) | Train Loss (NLL) | Test Loss (NLL)")
        for epoch in range(len(train_losses_ce)):
            print(f"{epoch + 1:5} | {train_losses_ce[epoch]:.4f} | {test_losses_ce[epoch]:.4f} | {train_losses_nll[epoch]:.4f} | {test_losses_nll[epoch]:.4f} ")