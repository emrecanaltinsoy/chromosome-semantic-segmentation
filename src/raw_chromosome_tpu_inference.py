import argparse
import yaml
import os
from glob import glob
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import skimage.io as io

import torch_xla
import torch_xla.core.xla_model as xm

from raw_chromosome_dataset import RawChromosomeDataset as Dataset
from loss import DiceLoss, evals

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

def main(args):
    args.model = 'preactivation_resunet'
    args.model_path = 'preactivation_resunet-20210331T2226'
    args.weight_num = 1
    args.images = "./datasets/raw_chromosome_data".format(Dataset.name)
    args.batch_size = 2
    args.test_results = True

    if args.model == 'unet':
        model = UNet(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes, init_features=32)
        net_name = UNet.net_name
    elif args.model == 'resunet':
        model = ResUNet(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes, init_features=32)
        net_name = 'resunet'
    elif args.model == 'preactivation_resunet':
        model = PreactResUNet(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes, init_features=32)
        net_name = 'preactivation_resunet'
    elif args.model == 'cenet':
        model = CE_Net(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = 'cenet'
    elif args.model == 'segnet':
        model = SegNet(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = 'segnet'
    elif args.model == 'nested_unet':
        model = UNet_Nested(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = 'nested_unet'
    elif args.model == 'attention_unet':
        model = AttU_Net(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = 'attention_unet'
    elif args.model == 'fcn_resnet101':
        model = FCN_ResNet101(in_channels=1, num_classes=3)
        net_name = 'fcn_resnet101'
    elif args.model == 'deeplabv3_resnet101':
        model = Deeplabv3_ResNet101(in_channels=1, num_classes=3)
        net_name = 'deeplabv3_resnet101'
    elif args.model == 'pspnet':
        model = PSPNet(num_classes=Dataset.num_classes, pretrained=False, backend='resnet101')
        net_name = 'pspnet'

    device = xm.xla_device()
    model.to(device)
    
    weights_dir = 'output/{}/{}/weights'.format(Dataset.name, args.model_path)
    print(weights_dir)
    model_name = glob(weights_dir + '/{}-{}*'.format(net_name, args.weight_num))[0] 
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)
    
    test_dir = 'output/{}/{}/test'.format(Dataset.name, args.model_path)

    model.eval()

    dsc = DiceLoss()

    evaluations = evals()
    evaluations_np = []

    total_dsc_loss = []

    loader = data_loaders(args)
    loaders = {"test": loader}

    start = time.time()
    print('clock started')

    test_img_num = 1

    for i, data in enumerate(loaders["test"], 0):
        x, y_true = data
        x, y_true = x.to(device, dtype=torch.float), y_true.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            y_pred = model(x)
            dsc_loss = dsc(y_pred, y_true)

            evaluations_ = evaluations(y_pred, y_true)
            evaluations_np += evaluations_
                
            total_dsc_loss.append(dsc_loss.item())

            if args.test_results:
                y_pred_np = y_pred.detach().cpu().numpy()
                x_np = x.detach().cpu().numpy()
                for img_num in range(y_pred_np.shape[0]):
                    for mask_num in range(y_pred_np.shape[1]):
                        io.imsave(os.path.join(test_dir,"{}_label{}.png".format(test_img_num,mask_num)), y_pred_np[img_num,mask_num,:,:])
                    for mask_num in range(x_np.shape[1]):
                        io.imsave(os.path.join(test_dir,"%d_image.png"%test_img_num), x_np[img_num,mask_num,:,:]*255)
                    test_img_num += 1

    end = time.time()
    print('{} seconds past'.format(end-start))

    evaluations_np = np.array(evaluations_np)
    with open('output/{}/{}/test-eval.npy'.format(Dataset.name, args.model_path), 'wb') as f:
        np.save(f, evaluations_np)

    mean_dsc_loss = float(np.mean(total_dsc_loss))
    mean_DSC = 1 - mean_dsc_loss
    metrics = {
        'mean_dsc_loss': mean_dsc_loss,
        'mean_DSC': mean_DSC,
    }
    with open('output/{}/{}/metrics.yaml'.format(Dataset.name, args.model_path), 'w') as fp:
        yaml.dump(metrics, fp)

    print(f'mean dsc loss={mean_dsc_loss}')
    print(f'mean DSC={mean_DSC}')

def data_loaders(args):
    dataset_test = datasets(args)
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
    )
    return loader_test

def datasets(args):
    test = Dataset(
        args,
        images_dir=args.images,
        subset="test",
        image_size=args.image_size,
        random_sampling=False,
    )
    return test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of G-banding chromosome Images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='preactivation_resunet',
        help="choose model",
    )
    parser.add_argument(
        "--weight-num",
        type=int,
        default=0,
        help="weight number for inference",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default='',
        help="path to weights file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="input batch size for training (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of workers for data loading (default: 1)",
    )
    parser.add_argument(
        "--images", type=str, default="./datasets/{}_data/train".format(Dataset.name), help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=Dataset.img_size,
        help="target input image size (default: 256x256)",
    )
    parser.add_argument(
        "--test-results",
        type=bool,
        default=False,
        help="Do you want to output the test results? (defauld: False)",
    )
    args = parser.parse_args()
    main(args)
