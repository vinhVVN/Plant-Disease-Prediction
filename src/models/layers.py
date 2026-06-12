import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

class SE_Module(nn.Module):
    def __init__(self, exp_size, squeeze_channels):
        super(SE_Module, self).__init__()
        self.fc1 = nn.Conv2d(exp_size, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, exp_size, kernel_size=1)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale

class Bneck(nn.Module):
    def __init__(self, in_channels, exp_size, out_channels, kernel_size, stride, use_se, nl_class):
        super(Bneck, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase
        if exp_size != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(exp_size),
                nl_class(inplace=True)
            ])

        # Depthwise phase
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=padding, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            nl_class(inplace=True)
        ])

        # Squeeze-and-Excite phase
        if use_se:
            squeeze_channels = _make_divisible(exp_size // 4, 8)
            layers.append(SE_Module(exp_size, squeeze_channels))

        # Projection phase
        layers.extend([
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
