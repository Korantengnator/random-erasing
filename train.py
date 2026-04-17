import torch                          
import torch.nn as nn                 
import torch.optim as optim           
import torchvision                   
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader   

import random
import numpy as np
import os
import json
from tqdm import tqdm                 


import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transforms.random_erasing import RandomErasing
from models.resnet import get_resnet



# Fix Random Seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Detect the Best Available Device
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple M4 GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")



# Set Up Data Loading
def get_data_loaders(use_random_erasing=False, batch_size=128):

    # Transforms applied to TRAIN images
    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    # If requested, add Random Erasing as the LAST transform.
    if use_random_erasing:
        train_transforms_list.append(
            RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3)
        )

    train_transform = transforms.Compose(train_transforms_list)

    # Transforms applied to TEST images
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # Download and load the datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=0
    )

    return train_loader, test_loader



# Train for One Epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  
    
    total_loss = 0
    correct = 0
    total = 0

    # tqdm wraps the loader to show a live progress bar
    for images, labels in tqdm(loader, desc="Training", leave=False):

        # Move data to the device 
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: compute the model's predictions for this batch
        outputs = model(images)

        # Calculate how wrong the model was (the loss).
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()   
        loss.backward()         
        optimizer.step()       

        # Track accuracy
        total_loss += loss.item()
        _, predicted = outputs.max(1)       
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy



# Evaluate on Test Set
def evaluate(model, loader, criterion, device):
    model.eval()  

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy, error_rate



# Main Training Function
def run_experiment(use_random_erasing=False, num_epochs=100, seed=42):

    experiment_name = "with_RE" if use_random_erasing else "baseline"
    print(f"\n{'='*50}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*50}\n")

    # Setup
    set_seed(seed)
    device = get_device()

    # Data
    train_loader, test_loader = get_data_loaders(
        use_random_erasing=use_random_erasing,
        batch_size=128
    )

    # Model — using ResNet-18 adapted for CIFAR-10
    model = get_resnet(num_layers=18, num_classes=10)
    model = model.to(device)

    # Loss function: CrossEntropyLoss is standard for classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: SGD with momentum, following the paper's training setup
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,           
        momentum=0.9,    
        weight_decay=1e-4 
    )

    # Learning rate scheduler: divides LR by 10 at epochs 150 and 225
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[150, 225],  # epochs where LR drops
        gamma=0.1               # multiply LR by 0.1 at each milestone
    )

    # Training Loop
    history = []  

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, test_error = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step() 

        # Save this epoch's results
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_error': test_error,
        })

        # Print a summary every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Test Error: {test_error:.2f}%"
            )

    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = f'results/{experiment_name}_history.json'
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save the trained model weights too
    torch.save(model.state_dict(), f'results/{experiment_name}_model.pth')

    final_error = history[-1]['test_error']
    print(f"\nFinal test error ({experiment_name}): {final_error:.2f}%")
    print(f"Results saved to {results_path}")

    return history



# Entry Point
if __name__ == "__main__":
    # Run 1: No Random Erasing (baseline)
    baseline_history = run_experiment(
        use_random_erasing=False,
        num_epochs=100,
        seed=42
    )

    # Run 2: With Random Erasing
    re_history = run_experiment(
        use_random_erasing=True,
        num_epochs=100,
        seed=42
    )

    # Print final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Baseline test error:       {baseline_history[-1]['test_error']:.2f}%")
    print(f"Random Erasing test error: {re_history[-1]['test_error']:.2f}%")
    improvement = baseline_history[-1]['test_error'] - re_history[-1]['test_error']
    print(f"Improvement:               {improvement:.2f}%")