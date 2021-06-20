import argparse
import yaml
import os
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_model_summary import summary

from raw_chromosome_dataset import RawChromosomeDataset as Dataset
from logger import Logger
from loss import DiceLoss, jaccard_distance_loss

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
    if args.model == "unet":
        model = UNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=args.init_features,
        )
        net_name = "unet"
    elif args.model == "resunet":
        model = ResUNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=args.init_features,
        )
        net_name = "resunet"
    elif args.model == "preactivation_resunet":
        model = PreactResUNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=args.init_features,
        )
        net_name = "preactivation_resunet"
    elif args.model == "cenet":
        model = CE_Net(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = "cenet"
    elif args.model == "segnet":
        model = SegNet(in_channels=Dataset.in_channels, num_classes=Dataset.num_classes)
        net_name = "segnet"
    elif args.model == "nested_unet":
        model = UNet_Nested(
            in_channels=Dataset.in_channels, num_classes=Dataset.num_classes
        )
        net_name = "nested_unet"
    elif args.model == "attention_unet":
        model = AttU_Net(
            in_channels=Dataset.in_channels, num_classes=Dataset.num_classes
        )
        net_name = "attention_unet"
    elif args.model == "fcn_resnet101":
        model = FCN_ResNet101(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            pretrained=args.pretrained,
        )
        net_name = "fcn_resnet101"
    elif args.model == "deeplabv3_resnet101":
        model = Deeplabv3_ResNet101(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            pretrained=args.pretrained,
        )
        net_name = "deeplabv3_resnet101"
    elif args.model == "pspnet":
        model = PSPNet(
            num_classes=Dataset.num_classes,
            pretrained=args.pretrained,
            backend="resnet101",
        )
        net_name = "pspnet"

    print(
        summary(
            model,
            torch.zeros((1, 1, Dataset.img_size[0], Dataset.img_size[1])),
            show_input=False,
            show_hierarchical=False,
        )
    )
    # print('number of parameters = ', sum(p.numel() for p in model.parameters()))
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    model.to(device)

    if args.weights == "":
        args.weights = "./output/{}/{}-{:%Y%m%dT%H%M}/weights".format(
            Dataset.name, model.net_name, Dataset.now
        )
    if args.logs == "":
        args.logs = "./output/{}/{}-{:%Y%m%dT%H%M}/logs".format(
            Dataset.name, model.net_name, Dataset.now
        )
    if args.test == "":
        args.test = "./output/{}/{}-{:%Y%m%dT%H%M}/test".format(
            Dataset.name, model.net_name, Dataset.now
        )

    make_dirs(args)
    save_args(args)
    best_validation_orig_loss = 1.0
    step = 0
    train_step = 0
    val_step = 0

    dsc = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    orig_loss_train = []
    orig_loss_valid = []
    dsc_loss_train = []
    dsc_loss_valid = []
    jaccard_distance_loss_train = []
    jaccard_distance_loss_valid = []
    phases = ["train", "valid"]

    epoch_step = 0

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    dsc_train = []
    dsc_valid = []

    for epoch in range(1, args.epochs + 1):

        for phase in phases:
            if phase == "train":
                model.train()
            elif phase == "valid":
                model.eval()

            for i, data in enumerate(loaders[phase], 0):

                x, y_true = data

                if phase == "train":
                    step += 1
                    epoch_step += 1

                x, y_true = x.to(device, dtype=torch.float), y_true.to(
                    device, dtype=torch.float
                )

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)

                    dsc_loss = dsc(y_pred, y_true)
                    dsc_ = 1 - dsc_loss

                    jaccard_distance = jaccard_distance_loss(y_pred, y_true)

                    orig_loss_name = "dsc_loss"
                    orig_loss = dsc_loss

                    if phase == "valid":
                        val_step += 1
                        orig_loss_valid.append(orig_loss.item())
                        dsc_loss_valid.append(dsc_loss.item())
                        jaccard_distance_loss_valid.append(jaccard_distance.item())

                        dsc_valid.append(dsc_.item())

                    if phase == "train":
                        orig_loss_train.append(orig_loss.item())
                        dsc_loss_train.append(dsc_loss.item())
                        jaccard_distance_loss_train.append(jaccard_distance.item())

                        dsc_train.append(dsc_.item())

                        orig_loss.backward()

                        optimizer.step()
                        if (step) % 4 == 0:
                            train_step += 1
                            print(
                                "Epoch={}/{}, step={}, orig_loss={}, DSC={}, dsc_loss={}, jaccard_distance={}".format(
                                    epoch,
                                    args.epochs,
                                    epoch_step,
                                    orig_loss.item(),
                                    dsc_.item(),
                                    dsc_loss.item(),
                                    jaccard_distance.item(),
                                )
                            )

            if phase == "train":
                mean_train_orig_loss = np.mean(orig_loss_train)
                mean_train_dsc_loss = np.mean(dsc_loss_train)
                train_jaccard_distance_loss = np.mean(jaccard_distance_loss_train)

                mean_train_dsc = np.mean(dsc_train)

                log_loss_summary(logger, mean_train_orig_loss, epoch, prefix="orig_")
                log_loss_summary(logger, mean_train_dsc_loss, epoch, prefix="dsc_")
                log_loss_summary(
                    logger,
                    train_jaccard_distance_loss,
                    epoch,
                    prefix="jaccard_distance_",
                )

                mean_train_dsc = []
                orig_loss_train = []
                dsc_loss_train = []
                jaccard_distance_loss_train = []

                dsc_train = []

            if phase == "valid":
                validation_orig_loss = np.mean(orig_loss_valid)
                validation_dsc_loss = np.mean(dsc_loss_valid)
                validation_jaccard_distance_loss = np.mean(jaccard_distance_loss_valid)

                log_loss_summary(
                    logger, validation_orig_loss, epoch, prefix="orig_val_"
                )
                log_loss_summary(logger, validation_dsc_loss, epoch, prefix="dsc_val_")
                log_loss_summary(
                    logger,
                    validation_jaccard_distance_loss,
                    epoch,
                    prefix="val_jaccard_distance_",
                )

                orig_loss_valid = []
                dsc_loss_valid = []
                jaccard_distance_loss_valid = []

                mean_validation_dsc = np.mean(dsc_valid)
                dsc_valid = []

                if validation_orig_loss < best_validation_orig_loss:
                    print("\n", f"saving weight into {args.weights}", "\n")
                    best_validation_orig_loss = validation_orig_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            args.weights,
                            "{}-{}-val_{}-{}.pt".format(
                                args.model,
                                epoch,
                                orig_loss_name,
                                best_validation_orig_loss,
                            ),
                        ),
                    )
                else:
                    print(
                        "\n",
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%weight is not saved%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
                        "\n",
                    )
            epoch_step = 0

        print(
            "Epoch={}/{}, orig_loss={}, val_orig_loss={}, DSC = {}, val_DSC = {}, dsc_loss={}, val_dsc_loss={}, jaccard_distance={}, val_jaccard_distance={}".format(
                epoch,
                args.epochs,
                mean_train_orig_loss,
                validation_orig_loss,
                mean_train_dsc,
                mean_validation_dsc,
                mean_train_dsc_loss,
                validation_dsc_loss,
                train_jaccard_distance_loss,
                validation_jaccard_distance_loss,
            )
        )


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
    )
    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        args,
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
    )
    valid = Dataset(
        args,
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def make_dirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.test, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def save_args(args):
    args_file = os.path.join(args.logs, "args.yaml")
    with open(args_file, "w") as fp:
        yaml.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of G-banding chromosome Images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="preactivation_resunet",
        help="choose model",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="is the backbone pretrained of not",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="input batch size for training (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="number of epochs to train (default: 40)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.0001)",
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
        default=0,
        help="number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--weights", type=str, default="", help="folder to save weights"
    )
    parser.add_argument("--test", type=str, default="", help="folder to save weights")
    parser.add_argument("--logs", type=str, default="", help="folder to save logs")
    parser.add_argument(
        "--images",
        type=str,
        default="./datasets/{}_data".format(Dataset.name),
        help="root folder with images",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=Dataset.img_size,
        help="target input image size (default: 480x640)",
    )
    parser.add_argument(
        "--init-features",
        type=int,
        default=32,
        help="init features for unet, resunet, preact-resunet",
    )
    args = parser.parse_args()
    main(args)
