import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101


class Resnet(nn.Module):
    def __init__(self, pretrained: bool = True, multiple_entity: bool = True):
        super(Resnet, self).__init__()

        self.r_index = -2 if multiple_entity else -1

        resnet = resnet101(pretrained=pretrained)

        self.net = nn.Sequential(*list(resnet.children())[:self.r_index])

    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x)
        return x.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2048)
