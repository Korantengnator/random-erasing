# This file just checks everything is connected correctly.
# It doesn't train anything — it just runs a few images through
# the model and makes sure nothing crashes.

import torch
import torchvision.transforms as transforms
import torchvision

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transforms.random_erasing import RandomErasing
from models.resnet import get_resnet

print("✓ All imports successful\n")

# --- Test 1: Does Random Erasing work? ---
print("Test 1: Random Erasing transform...")

# Make a fake image: 3 channels, 32x32 pixels, random values
fake_image = torch.rand(3, 32, 32)
print(f"  Image before erasing — min: {fake_image.min():.2f}, max: {fake_image.max():.2f}")

eraser = RandomErasing(p=1.0)  # p=1.0 means it ALWAYS erases (for testing)
erased = eraser(fake_image.clone())
print(f"  Image after erasing  — min: {erased.min():.2f}, max: {erased.max():.2f}")
print("  ✓ Random Erasing works\n")

# --- Test 2: Does the model build correctly? ---
print("Test 2: Building ResNet-18 for CIFAR-10...")
model = get_resnet(num_layers=18, num_classes=10)

# Count how many parameters the model has (just for interest)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Model built — {total_params:,} total parameters")
print("  ✓ Model built successfully\n")

# --- Test 3: Can the model process a batch of images? ---
print("Test 3: Running a batch of images through the model...")
fake_batch = torch.rand(4, 3, 32, 32)  # 4 images, 3 channels, 32x32
output = model(fake_batch)
print(f"  Input shape:  {list(fake_batch.shape)}  (4 images)")
print(f"  Output shape: {list(output.shape)}  (4 images × 10 class scores)")
print("  ✓ Forward pass works\n")

# --- Test 4: Does data loading work? ---
print("Test 4: Loading CIFAR-10 data (will download if first time)...")
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)
print(f"  Test set size: {len(dataset)} images")
print("  ✓ Data loading works\n")

print("=" * 40)
print("All tests passed! You are ready to train.")
print("=" * 40)