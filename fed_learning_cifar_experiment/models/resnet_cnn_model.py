import torch
import torch.nn as nn
from typing import Type, Callable, List, Optional

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def _make_layer(block: Type[BasicBlock], in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(in_planes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(block(planes * block.expansion, planes))
    return nn.Sequential(*layers)

class TinyResNet18(nn.Module):
    """
    CIFAR-10 friendly ResNet-18 layout (2-2-2-2) with narrow channels.
    Default width yields ~0.27M params (close to the paper’s lightweight model).
    """
    def __init__(self, num_classes: int = 10, base_width: int = 8):
        """
        base_width controls total params:
          8 -> ~0.27M, 12 -> ~0.6M, 16 -> ~1.05M (approx).
        """
        super().__init__()
        self.in_planes = base_width

        # CIFAR stem: 3x3, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.in_planes)
        self.relu  = nn.ReLU(inplace=True)

        # ResNet-18 stages; downsample at the start of layers 2–4
        self.layer1 = _make_layer(BasicBlock, self.in_planes, base_width, blocks=2, stride=1)
        self.layer2 = _make_layer(BasicBlock, base_width,    base_width*2, blocks=2, stride=2)
        self.layer3 = _make_layer(BasicBlock, base_width*2,  base_width*4, blocks=2, stride=2)
        self.layer4 = _make_layer(BasicBlock, base_width*4,  base_width*8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width*8, num_classes)

        # Kaiming init (like torchvision)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.05
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def tiny_resnet18(num_classes: int = 10, base_width: int = 8) -> TinyResNet18:
    return TinyResNet18(num_classes=num_classes, base_width=base_width)

if __name__ == "__main__":
    # quick sanity check
    model = tiny_resnet18()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
    total = sum(p.numel() for p in model.parameters())
    print("Params:", total)
