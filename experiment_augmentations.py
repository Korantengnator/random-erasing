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


def get_loaders(use_rf=False, use_rc=False, use_re=False, batch_size=128):
    """
    Builds a data loader with any combination of the three augmentations.

    use_rf : Random Horizontal Flipping
    use_rc : Random Cropping (pad then crop)
    use_re : Random Erasing
    """

    train_transforms_list = []

    # Random Cropping
    if use_rc:
        train_transforms_list.append(transforms.RandomCrop(32, padding=4))

    # Random Flipping 
    if use_rf:
        train_transforms_list.append(transforms.RandomHorizontalFlip())

    # ToTensor and Normalize (must come before Random Erasing)
    train_transforms_list.append(transforms.ToTensor())
    train_transforms_list.append(
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    )

    # Random Erasing (must come after ToTensor and Normalize)
    if use_re:
        train_transforms_list.append(
            RandomErasing(p=0.5, sl=0.02, sh=0.4, r1=0.3, fill_mode='random')
        )

    train_transform = transforms.Compose(train_transforms_list)

    # Test transform never has augmentation
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

    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=128,
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


def run_augmentation_experiment(name, use_rf, use_rc, use_re,
                                num_epochs=100, seed=42):
    print(f"\n{'='*50}")
    print(f"Running: {name}")
    print(f"  Random Flip: {use_rf} | Random Crop: {use_rc} | Random Erase: {use_re}")
    print(f"{'='*50}")

    set_seed(seed)
    device = get_device()
    train_loader, test_loader = get_loaders(use_rf, use_rc, use_re)

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
    safe_name = name.replace(' ', '_').replace('+', 'and')
    with open(f'results/aug_{safe_name}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    final_error = history[-1]['test_error']
    print(f"Final test error: {final_error:.2f}%")
    return final_error


def plot_augmentation_comparison(results):
    """Bar chart of all 8 combinations — mirrors Table 5 in the paper."""
    names  = list(results.keys())
    errors = list(results.values())

    # Colour the bars: highlight the all-three combination in green
    colours = ['#4CAF50' if 'RF+RC+RE' in n else '#2196F3' for n in names]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(names, errors, color=colours, width=0.6)
    plt.ylabel('Test Error (%)')
    plt.title('Test Error for All Augmentation Combinations (CIFAR-10, ResNet-18)')
    plt.xticks(rotation=20, ha='right')
    plt.ylim(min(errors) - 1, max(errors) + 2)

    for bar, error in zip(bars, errors):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'{error:.2f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/augmentation_comparison.png', dpi=150,
                bbox_inches='tight')
    print("Chart saved to results/augmentation_comparison.png")
    plt.show()


if __name__ == "__main__":

    # Define the 8 combinations of augmentations to test. 
    experiments = [
        ("None",        False, False, False),
        ("RF",          True,  False, False),
        ("RC",          False, True,  False),
        ("RE",          False, False, True ),
        ("RF+RC",       True,  True,  False),
        ("RF+RE",       True,  False, True ),
        ("RC+RE",       False, True,  True ),
        ("RF+RC+RE",    True,  True,  True ),
    ]

    results = {}
    for name, use_rf, use_rc, use_re in experiments:
        results[name] = run_augmentation_experiment(
            name, use_rf, use_rc, use_re, num_epochs=100, seed=42
        )

    # Print summary table
    print("\n" + "="*40)
    print(f"{'AUGMENTATION COMPARISON':^40}")
    print("="*40)
    print(f"{'Method':<15} {'Test Error':>10}")
    print("-"*40)
    for name, error in results.items():
        marker = " ← best" if error == min(results.values()) else ""
        print(f"{name:<15} {error:>9.2f}%{marker}")
    print("="*40)

    plot_augmentation_comparison(results)