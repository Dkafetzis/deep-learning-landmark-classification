import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights


# Well use resnset18, 50 and 152 for this
def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    # Get the requested architecture
    if hasattr(models, model_name):
        if model_name == "resnet18":
            model_transfer = getattr(models, model_name)(weights=ResNet18_Weights.IMAGENET1K_V1)
        if model_name == "resnet50":
            model_transfer = getattr(models, model_name)(weights=ResNet50_Weights.IMAGENET1K_V1)
        if model_name == "resnet152":
            model_transfer = getattr(models, model_name)(weights=ResNet152_Weights.IMAGENET1K_V1)

    # Freeze all parameters in the model
    frozen_parameters = []

    for p in model_transfer.parameters():
        if p.requires_grad:
            p.requires_grad = False
            frozen_parameters.append(p)

    print(f"Froze {len(frozen_parameters)} groups of parameters")

    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs = model_transfer.fc.in_features

    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer
