# ─────────────────────────────────────────────
# SECTION 1: Imports
# ─────────────────────────────────────────────
# These are all the libraries we need.
# Think of imports like gathering your tools before starting a job.

import torch                          # the main deep learning library
import torch.nn as nn                 # neural network building blocks
import torch.optim as optim           # optimizers (how the model learns)
import torchvision                    # datasets and image utilities
import torchvision.transforms as transforms  # image transformations
from torch.utils.data import DataLoader     # handles batching images efficiently

import random
import numpy as np
import os
import json
from tqdm import tqdm                 # shows a nice progress bar during training

# Our own files
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transforms.random_erasing import RandomErasing
from models.resnet import get_resnet


# ─────────────────────────────────────────────
# SECTION 2: Fix Random Seeds
# ─────────────────────────────────────────────
# The assignment requires that running the code twice gives the same result.
# Neural networks use randomness (shuffling data, initializing weights, etc.)
# "Seeding" fixes that randomness so results are reproducible.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# SECTION 3: Detect the Best Available Device
# ─────────────────────────────────────────────
# "Device" = where computations happen.
# Your M4 Mac has a GPU accessed via "mps" (Metal Performance Shaders).
# If that's not available, we fall back to regular CPU.

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


# ─────────────────────────────────────────────
# SECTION 4: Set Up Data Loading
# ─────────────────────────────────────────────
# This function downloads CIFAR-10 and prepares two versions:
# - one WITH Random Erasing (for the experimental run)
# - one WITHOUT (for the baseline run)
#
# "use_random_erasing=True" acts like a switch between the two.

def get_data_loaders(use_random_erasing=False, batch_size=128):

    # --- Transforms applied to TRAINING images ---
    # Transforms are applied one after another, in order, like an assembly line.

    train_transforms_list = [
        # 1. Pad the 32x32 image to 40x40, then randomly crop back to 32x32.
        #    This simulates the object being in slightly different positions.
        transforms.RandomCrop(32, padding=4),

        # 2. Randomly flip the image horizontally (like a mirror).
        #    A car facing left is still a car — this helps the model learn that.
        transforms.RandomHorizontalFlip(),

        # 3. Convert the image from a PIL Image to a PyTorch tensor.
        #    Also scales pixel values from [0, 255] to [0.0, 1.0].
        transforms.ToTensor(),

        # 4. Normalize each colour channel using CIFAR-10's mean and std.
        #    This centres the data around 0, which helps the model train faster.
        #    These specific numbers are the known statistics of CIFAR-10.
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    # If requested, add Random Erasing as the LAST transform.
    # It must come after ToTensor() because it operates on tensors, not images.
    if use_random_erasing:
        train_transforms_list.append(
            RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3)
        )

    train_transform = transforms.Compose(train_transforms_list)

    # --- Transforms applied to TEST images ---
    # No augmentation at test time! We want a fair, consistent evaluation.
    # We only convert to tensor and normalize (same normalization as training).
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    # --- Download and load the datasets ---
    # download=True means PyTorch will fetch CIFAR-10 automatically if needed.
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

    # --- Create DataLoaders ---
    # A DataLoader groups images into "batches" so the model processes many at once.
    # batch_size=128 means the model sees 128 images before updating its weights.
    # shuffle=True means training images are served in a different order each epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0   # 0 is safest on Mac; avoids multiprocessing issues
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # no need to shuffle test data
        num_workers=0
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
# SECTION 5: Train for One Epoch
# ─────────────────────────────────────────────
# One "epoch" = the model has seen every training image once.
# This function handles one full pass through the training data.

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # puts model in training mode (enables dropout etc.)
    
    total_loss = 0
    correct = 0
    total = 0

    # tqdm wraps the loader to show a live progress bar
    for images, labels in tqdm(loader, desc="Training", leave=False):

        # Move data to the device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward pass ---
        # Feed images through the model to get predictions (called "logits").
        # Logits are raw scores for each class — not probabilities yet.
        outputs = model(images)

        # Calculate how wrong the model was (the loss).
        # CrossEntropyLoss is the standard loss for classification.
        loss = criterion(outputs, labels)

        # --- Backward pass ---
        # This is where the model learns. Three steps:
        optimizer.zero_grad()   # 1. clear old gradients (must do this each step)
        loss.backward()         # 2. compute new gradients (how to adjust weights)
        optimizer.step()        # 3. update the weights using the gradients

        # --- Track accuracy ---
        total_loss += loss.item()
        _, predicted = outputs.max(1)       # pick the class with highest score
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# SECTION 6: Evaluate on Test Set
# ─────────────────────────────────────────────
# Runs the model on the test set (no learning happens here).
# Returns the test error rate — this is the number we compare to the paper.

def evaluate(model, loader, criterion, device):
    model.eval()  # puts model in evaluation mode (disables dropout etc.)

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # no need to track gradients during evaluation
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


# ─────────────────────────────────────────────
# SECTION 7: Main Training Function
# ─────────────────────────────────────────────
# This ties everything together and runs a full experiment.

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
        lr=0.1,           # starting learning rate (paper uses 0.1)
        momentum=0.9,     # helps training move smoothly
        weight_decay=1e-4 # L2 regularization to prevent overfitting
    )

    # Learning rate scheduler: divides LR by 10 at epochs 150 and 225
    # This is exactly what the paper describes
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[150, 225],  # epochs where LR drops
        gamma=0.1               # multiply LR by 0.1 at each milestone
    )

    # --- Training Loop ---
    history = []  # we'll save results here to plot later

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, test_error = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()  # update the learning rate

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

    # --- Save results ---
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


# ─────────────────────────────────────────────
# SECTION 8: Entry Point
# ─────────────────────────────────────────────
# This runs when you execute: python train.py
# It runs the baseline first, then the Random Erasing experiment.

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