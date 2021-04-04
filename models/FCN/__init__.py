import torch
from torch import nn
from torchvision import models

class FCN_ResNet50(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, pretrained=False):
        super(FCN_ResNet50, self).__init__()
        self.fcn = models.segmentation.fcn_resnet50(pretrained=pretrained, num_classes=num_classes)

    net_name = 'fcn_resnet50'

    def forward(self, x):
        if x.shape[1] != 3:
            x = torch.cat((x, x, x), 1)

        x = self.fcn(x)

        return torch.sigmoid(x['out'])

class FCN_ResNet101(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, pretrained=False):
        super(FCN_ResNet101, self).__init__()
        self.fcn = models.segmentation.fcn_resnet101(pretrained=pretrained, num_classes=num_classes)

    net_name = 'fcn_resnet101'

    def forward(self, x):
        if x.shape[1] != 3:
            x = torch.cat((x, x, x), 1)

        x = self.fcn(x)

        return torch.sigmoid(x['out'])