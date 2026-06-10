import torch
import torch.nn as nn
from src.models.layers import Bneck, Hswish, _make_divisible
from torchinfo import summary

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=38):
        super(MobileNetV3Small, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = Hswish(inplace=True)

        # 11 Inverted Residual Bottlenecks matching the Table
        # [kernel_size, exp_size, out_channels, use_se, activation, stride]
        bneck_config = [
            [3, 16, 16, True, nn.ReLU, 2],
            [3, 72, 24, False, nn.ReLU, 2],
            [3, 88, 24, False, nn.ReLU, 1],
            [5, 96, 40, True, Hswish, 2],
            [5, 240, 40, True, Hswish, 1],
            [5, 240, 40, True, Hswish, 1],
            [5, 120, 48, True, Hswish, 1],
            [5, 144, 48, True, Hswish, 1],
            [5, 288, 96, True, Hswish, 2],
            [5, 576, 96, True, Hswish, 1],
            [5, 576, 96, True, Hswish, 1],
        ]

        layers = []
        in_channels = 16
        for k, exp_size, c, se, nl, s in bneck_config:
            layers.append(Bneck(in_channels, exp_size, c, k, s, se, nl))
            in_channels = c
        self.bnecks = nn.Sequential(*layers)

        # Last Conv (Projection before pooling)
        self.conv2 = nn.Conv2d(in_channels, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = Hswish(inplace=True)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            Hswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hs1(x)

        x = self.bnecks(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hs2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    
    # 1. Initialize model
    model = MobileNetV3Small(num_classes=38)
    
    # 2. Test Output Shape Mathematically
    # Tensor representation: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    out = model(dummy_input)
    print(f"Sanity Check - Expected Shape: (1, 38) | Actual Shape: {out.shape}")
    assert out.shape == (1, 38), "Output shape mismatch!"
    
    # 3. Test Total Trainable Parameters (Checking against paper's ~1.54M metric)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # 4. Detailed Architectural Summary
    summary(model, input_size=(1, 3, 224, 224))
