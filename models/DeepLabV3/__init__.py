import torch
from torch import nn
from torchvision import models

class Deeplabv3_ResNet50(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, pretrained=False):
        super(Deeplabv3_ResNet50, self).__init__()
        self.deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, num_classes=num_classes)

    net_name = 'deeplabv3_resnet50'

    def forward(self, x):
        if x.shape[1] != 3:
            x = torch.cat((x, x, x), 1)

        x = self.deeplabv3(x)

        return torch.sigmoid(x['out'])

class Deeplabv3_ResNet101(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, pretrained=False):
        super(Deeplabv3_ResNet101, self).__init__()
        self.deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, num_classes=num_classes)

    net_name = 'deeplabv3_resnet101'

    def forward(self, x):
        if x.shape[1] != 3:
            x = torch.cat((x, x, x), 1)

        x = self.deeplabv3(x)

        return torch.sigmoid(x['out'])