import torch.nn as nn
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=38, dropout_rate=0.2, pretrained=True):
        super(EfficientNetB0, self).__init__()
        
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.model = models.efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def freeze_backbone(model):
    for param in model.model.features.parameters():
        param.requires_grad = False

    for param in model.model.classifier.parameters():
        param.requires_grad = True

    return model

def unfreeze_backbone(model, unfreeze_all=False):
    if unfreeze_all:
        for param in model.model.features.parameters():
            param.requires_grad = True
    else:
        for param in model.model.features[-2:].parameters():
            param.requires_grad = True

    return model
