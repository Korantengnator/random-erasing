import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import os
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transforms.random_erasing import RandomErasing
from models.resnet import get_resnet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_loaders(fill_mode=None, batch_size=128):
    """
    fill_mode=None means no Random Erasing (baseline).
    fill_mode='random'/'mean'/'zero'/'max' enables the chosen fill strategy.
    """

    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    if fill_mode is not None:
        train_transforms_list.append(
            RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3, fill_mode=fill_mode)
        )

    train_transform = transforms.Compose(train_transforms_list)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total, total_loss = 0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    return total_loss / len(loader), accuracy, 100.0 - accuracy


def run_fill_experiment(fill_mode, num_epochs=100, seed=42):
    name = fill_mode if fill_mode is not None else 'baseline'
    print(f"\n{'='*50}")
    print(f"Running: {name.upper()}")
    print(f"{'='*50}")

    set_seed(seed)
    device = get_device()
    train_loader, test_loader = get_loaders(fill_mode=fill_mode)

    model = get_resnet(num_layers=18, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 225], gamma=0.1
    )

    history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, test_error = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()
        history.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_error': test_error,
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: {train_acc:.2f}% | "
                  f"Test Error: {test_error:.2f}%")

    os.makedirs('results', exist_ok=True)
    with open(f'results/fill_{name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    final_error = history[-1]['test_error']
    print(f"Final test error ({name}): {final_error:.2f}%")
    return final_error


def plot_fill_comparison(results):
    """Bar chart comparing the four fill modes — mirrors Table 3 in the paper."""
    labels = list(results.keys())
    errors = list(results.values())
    colours = ['#888888', '#2196F3', '#FF9800', '#F44336', '#4CAF50']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, errors, color=colours[:len(labels)], width=0.5)
    plt.ylabel('Test Error (%)')
    plt.title('Test Error by Erasing Fill Mode (CIFAR-10, ResNet-18)')
    plt.ylim(min(errors) - 1, max(errors) + 1)

    # Add the error value on top of each bar
    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f'{error:.2f}%',
                 ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('results/fill_comparison.png', dpi=150, bbox_inches='tight')
    print("\nBar chart saved to results/fill_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Define the fill modes to test. None means no Random Erasing (baseline). 
    experiments = {
        'Baseline':  None,
        'RE-R (Random)': 'random',
        'RE-M (Mean)':   'mean',
        'RE-0 (Zero)':   'zero',
        'RE-255 (Max)':  'max',
    }

    results = {}
    for name, fill_mode in experiments.items():
        results[name] = run_fill_experiment(fill_mode, num_epochs=100, seed=42)

    # Print summary table
    print("\n" + "="*45)
    print(f"{'FILL VALUE COMPARISON':^45}")
    print("="*45)
    print(f"{'Method':<20} {'Test Error':>10}")
    print("-"*45)
    for name, error in results.items():
        print(f"{name:<20} {error:>9.2f}%")
    print("="*45)

    plot_fill_comparison(results)