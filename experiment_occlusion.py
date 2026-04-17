import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.resnet import get_resnet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def add_occlusion(images, occlusion_fraction):
    """
    Adds a square occlusion patch to the centre of each image in a batch.

    occlusion_fraction: how much of the image the patch covers (e.g. 0.3 = 30%)
    
    We use a fixed centre patch (not random) so results are consistent
    across different occlusion levels and both models see identical patches.
    """
    if occlusion_fraction == 0:
        return images

    occluded = images.clone()
    _, _, H, W = images.shape  

    # Calculate the patch size based on the fraction
    patch_size = int((occlusion_fraction ** 0.5) * H)

    # Place the patch in the centre of the image
    start = (H - patch_size) // 2
    end   = start + patch_size

    # Fill the patch with zeros (black rectangle)
    occluded[:, :, start:end, start:end] = 0.0

    return occluded


def evaluate_with_occlusion(model, loader, device, occlusion_fraction):
    """
    Evaluates the model on the test set with a given occlusion level applied.
    Returns the test error rate.
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Apply occlusion patch before passing to the model
            images = add_occlusion(images, occlusion_fraction)

            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    error_rate = 100.0 * (1 - correct / total)
    return error_rate


def load_trained_model(model_path, device):
    """Loads a previously saved model from disk."""
    model = get_resnet(num_layers=18, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":

    set_seed(42)
    device = get_device()

    # Check if saved models exist 
    baseline_path = 'results/baseline_model.pth'
    re_path       = 'results/with_RE_model.pth'

    if not os.path.exists(baseline_path) or not os.path.exists(re_path):
        print("ERROR: Could not find saved models.")
        print("Make sure you have run train.py first.")
        print(f"Looking for: {baseline_path}")
        print(f"Looking for: {re_path}")
        exit()

    print("Loading trained models...")
    baseline_model = load_trained_model(baseline_path, device)
    re_model       = load_trained_model(re_path, device)
    print("Models loaded successfully.\n")

    # Load the test set (no augmentation, just normalisation) 
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    # Define the occlusion levels to test (0% to 50% of the image area)
    occlusion_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Run evaluation at each occlusion level 
    print("Evaluating both models at each occlusion level...\n")

    baseline_errors = []
    re_errors       = []

    for level in occlusion_levels:
        b_error = evaluate_with_occlusion(
            baseline_model, test_loader, device, level
        )
        r_error = evaluate_with_occlusion(
            re_model, test_loader, device, level
        )

        baseline_errors.append(b_error)
        re_errors.append(r_error)

        print(f"Occlusion {int(level*100):3d}% | "
              f"Baseline Error: {b_error:.2f}% | "
              f"RE Error: {r_error:.2f}%")

    #  Print summary table 
    print("\n" + "="*55)
    print(f"{'OCCLUSION ROBUSTNESS RESULTS':^55}")
    print("="*55)
    print(f"{'Occlusion':<15} {'Baseline Error':>15} {'RE Error':>15}")
    print("-"*55)
    for level, b_err, r_err in zip(occlusion_levels,
                                    baseline_errors, re_errors):
        diff = b_err - r_err
        marker = f"  (RE better by {diff:.2f}%)" if diff > 0 else ""
        print(f"{int(level*100):>6}%{'':<8} {b_err:>13.2f}% "
              f"{r_err:>13.2f}%{marker}")
    print("="*55)

    # Plot the results 
    plt.figure(figsize=(9, 5))

    plt.plot(
        [l * 100 for l in occlusion_levels],
        baseline_errors,
        label='Baseline',
        color='steelblue',
        linewidth=2,
        marker='o',
        markersize=6
    )
    plt.plot(
        [l * 100 for l in occlusion_levels],
        re_errors,
        label='Random Erasing',
        color='darkorange',
        linewidth=2,
        marker='s',
        markersize=6
    )

    plt.xlabel('Occlusion Level (% of image area blocked)', fontsize=12)
    plt.ylabel('Test Error (%)', fontsize=12)
    plt.title('Robustness to Occlusion — Baseline vs Random Erasing\n'
              '(CIFAR-10, ResNet-18)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/occlusion_robustness.png', dpi=150,
                bbox_inches='tight')
    print("\nPlot saved to results/occlusion_robustness.png")
    plt.show()