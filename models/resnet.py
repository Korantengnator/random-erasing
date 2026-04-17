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
    layer_map = {
        18:  models.resnet18,
        34:  models.resnet34,
        50:  models.resnet50,
        101: models.resnet101,
    }

    # Default to ResNet-18 if the requested number of layers isn't available.
    if num_layers not in layer_map:
        print(f"Note: ResNet-{num_layers} not in torchvision. Using ResNet-18 instead.")
        num_layers = 18

    # Build the model
    model = layer_map[num_layers](weights=None)

    # Modify the first convolutional layer to better suit CIFAR-10's small images.
    model.conv1 = nn.Conv2d(
        in_channels=3,    
        out_channels=64, 
        kernel_size=3,    
        stride=1,         
        padding=1,        
        bias=False
    )

    # Remove the initial max pooling layer.
    model.maxpool = nn.Identity()  

    # Final classification layer.
    num_features = model.fc.in_features  
    model.fc = nn.Linear(num_features, num_classes)

    return model