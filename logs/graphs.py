import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.UNet import UNet
from models.ResUNet import ResUNet
from models.PreactivationResUNet import PreactResUNet
from models.CENet import CE_Net
from models.Segnet import SegNet
from models.AttentionUnet import AttU_Net
from models.FCN import FCN_ResNet101
from models.Unet_nested import UNet_Nested
from models.DeepLabV3 import Deeplabv3_ResNet101
from models.PSPNet import PSPNet

models = ['unet', 'resunet', 'proposed_cnn', 'cenet', 'segnet', 'nested_unet', 'attention_unet', 'fcn_resnet101', 'deeplabv3_resnet101', 'pspnet']

for m in models:
    if m == 'unet':
        model = UNet(in_channels=1, num_classes=3, init_features=32)
    elif m == 'resunet':
        model = ResUNet(in_channels=1, num_classes=3, init_features=32)
    elif m == 'proposed_cnn':
        model = PreactResUNet(in_channels=1, num_classes=3, init_features=32)
    elif m == 'cenet':
        model = CE_Net(in_channels=1, num_classes=3)
    elif m == 'segnet':
        model = SegNet(in_channels=1, num_classes=3)
    elif m == 'nested_unet':
        model = UNet_Nested(in_channels=1, num_classes=3)
    elif m == 'attention_unet':
        model = AttU_Net(in_channels=1, num_classes=3)
    elif m == 'fcn_resnet101':
        model = FCN_ResNet101(in_channels=1, num_classes=3, pretrained=False)
    elif m == 'deeplabv3_resnet101':
        model = Deeplabv3_ResNet101(in_channels=1, num_classes=3, pretrained=False)
    elif m == 'pspnet':
        model = PSPNet(num_classes=3, pretrained=False, backend='resnet101')

    writer = SummaryWriter(f"logs/graphs/{m}")

    data = torch.zeros([2, 1, 480, 640])
    writer.add_graph(model, data)
    writer.close()
    