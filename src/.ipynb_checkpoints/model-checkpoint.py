import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_model(num_classes=10):
    # Load pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model

