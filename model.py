import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            models.resnet50()
        )
        self.out = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.out(self.layer(x))
        out = torch.squeeze(out, dim=1)
        return out


if __name__ == '__main__':
    net = ResNet()
    a = torch.randn(3, 3, 304, 304)
    print(net(a))

