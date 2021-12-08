import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            models.mobilenet_v3_large()
        )
        self.out = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.out(self.layer(x))
        out = torch.squeeze(out, dim=1)
        return out


if __name__ == '__main__':
    net = ResNet()
    a = torch.rand(10, 3, 400, 400)
    print(net(a))

