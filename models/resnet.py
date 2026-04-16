import torch.nn as nn
import torchvision.models as models


def get_resnet(num_layers=20, num_classes=10):
    """
    Returns a ResNet model configured for CIFAR-10.

    Args:
        num_layers  : how deep the network is (20, 32, 44, 56, or 110)
        num_classes : number of output categories (10 for CIFAR-10)
    
    Returns:
        A PyTorch model ready to be trained.
    """

    # Map the number of layers to PyTorch's built-in ResNet versions.
    # ResNet-18 is the closest small ResNet available in torchvision.
    # For CIFAR-10, ResNet-18 is a very reasonable stand-in for ResNet-20.
    layer_map = {
        18:  models.resnet18,
        34:  models.resnet34,
        50:  models.resnet50,
        101: models.resnet101,
    }

    # Default to ResNet-18 if someone asks for a CIFAR-style depth like 20
    if num_layers not in layer_map:
        print(f"Note: ResNet-{num_layers} not in torchvision. Using ResNet-18 instead.")
        num_layers = 18

    # Build the model — pretrained=False means random initial weights
    # (we train from scratch, just like the paper)
    model = layer_map[num_layers](weights=None)

    # CIFAR-10 images are 32x32 pixels — much smaller than ImageNet (224x224).
    # The default ResNet first layer uses a 7x7 kernel designed for big images.
    # We swap it for a smaller 3x3 kernel so it works well on tiny CIFAR images.
    model.conv1 = nn.Conv2d(
        in_channels=3,    # 3 colour channels (R, G, B)
        out_channels=64,  # number of filters (keep the same as original)
        kernel_size=3,    # smaller kernel for small images
        stride=1,         # no aggressive downsampling at the start
        padding=1,        # keeps the spatial size the same after convolution
        bias=False
    )

    # Also remove the max pooling layer that follows conv1 in the original ResNet.
    # Again, that pooling was designed for large images — CIFAR doesn't need it.
    model.maxpool = nn.Identity()  # Identity just passes input through unchanged

    # Finally, replace the last layer (the classifier) to output 10 scores
    # instead of ImageNet's 1000 classes.
    # model.fc is the "fully connected" final layer.
    num_features = model.fc.in_features  # how many inputs the final layer takes
    model.fc = nn.Linear(num_features, num_classes)

    return model