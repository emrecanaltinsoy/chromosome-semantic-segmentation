import argparse
import os
import inspect
import sys
import csv
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import convex_hull_image

import torch
from torch.utils.data import DataLoader
import skimage.io as io

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from segmentation_dataset import RawChromosomeDataset as Dataset
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


def inference(args):
    if args.model == "unet":
        model = UNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=32,
        )
        net_name = UNet.net_name
    elif args.model == "resunet":
        model = ResUNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=32,
        )
        net_name = "resunet"
    elif args.model == "preactivation_resunet":
        model = PreactResUNet(
            in_channels=Dataset.in_channels,
            num_classes=Dataset.num_classes,
            init_features=32,
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
        model = FCN_ResNet101(in_channels=1, num_classes=3)
        net_name = "fcn_resnet101"
    elif args.model == "deeplabv3_resnet101":
        model = Deeplabv3_ResNet101(in_channels=1, num_classes=3)
        net_name = "deeplabv3_resnet101"
    elif args.model == "pspnet":
        model = PSPNet(
            num_classes=Dataset.num_classes, pretrained=False, backend="resnet101"
        )
        net_name = "pspnet"

    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    model.to(device)

    weights_dir = "output/{}/{}/weights".format(Dataset.name, args.weight_path)
    model_name = glob(weights_dir + "/{}-{}*".format(net_name, args.weight_num))[0]
    state_dict = torch.load(model_name, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    for k, loader in loaders.items():
        test_img_num = 1
        save_dir = "output/{}/{}/{}".format(Dataset.name, args.weight_path, k)
        os.makedirs(save_dir, exist_ok=True)

        for i, data in enumerate(loader, 0):
            x, y_true = data
            x, y_true = x.to(device, dtype=torch.float), y_true.to(
                device, dtype=torch.float
            )

            with torch.set_grad_enabled(False):
                y_pred = model(x)
                y_pred_np = y_pred.detach().cpu().numpy()
                y_true_np = y_true.detach().cpu().numpy()

                io.imsave(
                    os.path.join(save_dir, f"{test_img_num}_label1.png"),
                    y_pred_np[0, 1, :, :],
                )
                io.imsave(
                    os.path.join(save_dir, f"{test_img_num}_label2.png"),
                    y_pred_np[0, 2, :, :],
                )

                io.imsave(
                    os.path.join(save_dir, f"{test_img_num}_label1_orig.png"),
                    y_true_np[0, 1, :, :],
                )

                test_img_num += 1


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        drop_last=True,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        drop_last=False,
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


def extract_features_and_save(args):
    for folder_name in ["train", "valid"]:
        images_dir = f"output/{Dataset.name}/{args.weight_path}/{folder_name}"
        num_images = len(glob(f"{images_dir}/*.png")) // 3

        with open(
            args.save_dir + folder_name + "_data.csv",
            mode="w",
        ) as csv_file:
            fieldnames = [
                "area",
                "conv_hull_area",
                "average_label1_pixel",
                "average_label2_pixel",
                "obj_id",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(1, num_images + 1):
                print(f"Processing {folder_name} image {i}")
                label1_dir = f"{images_dir}/{i}_label1.png"
                label2_dir = f"{images_dir}/{i}_label2.png"
                origmask_dir = f"{images_dir}/{i}_label1_orig.png"

                Im_label1 = cv2.imread(label1_dir)
                Im_label2 = cv2.imread(label2_dir)
                Mask_orig = cv2.imread(origmask_dir)[:, :, 0]
                I_label1_gray = cv2.cvtColor(Im_label1, cv2.COLOR_BGR2GRAY)
                I_label2_gray = cv2.cvtColor(Im_label2, cv2.COLOR_BGR2GRAY)

                _, I_label1_bin = cv2.threshold(
                    I_label1_gray, 0, 255, cv2.THRESH_BINARY
                )
                _, I_label2_bin = cv2.threshold(
                    I_label2_gray, 255 * 0.99, 255, cv2.THRESH_BINARY
                )

                I_bin = I_label1_bin - I_label2_bin

                kernel = np.ones((3, 3), np.uint8)
                I_bin = cv2.morphologyEx(I_bin, cv2.MORPH_OPEN, kernel)

                connectivity = 8
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    I_bin, connectivity, cv2.CV_32S
                )

                areas = []

                color_black = (0, 0, 0)
                w_max = 0
                h_max = 0
                for num_labels_i in range(1, num_labels):
                    area = stats[num_labels_i, cv2.CC_STAT_AREA]
                    w = stats[num_labels_i, cv2.CC_STAT_WIDTH]
                    h = stats[num_labels_i, cv2.CC_STAT_HEIGHT]
                    w_max = np.maximum(w_max, w)
                    h_max = np.maximum(h_max, h)
                    if area > 10:
                        areas.append(area)

                obj_id_list = []
                area_list = []
                conv_hull_area_list = []

                label1_pixel_list = []
                label2_pixel_list = []

                for num_labels_i in range(1, num_labels):
                    label_obj = labels == num_labels_i
                    x = stats[num_labels_i, cv2.CC_STAT_LEFT]
                    y = stats[num_labels_i, cv2.CC_STAT_TOP]
                    w = stats[num_labels_i, cv2.CC_STAT_WIDTH]
                    h = stats[num_labels_i, cv2.CC_STAT_HEIGHT]
                    area = stats[num_labels_i, cv2.CC_STAT_AREA]

                    if area > 10:
                        obj = label_obj[y : y + h, x : x + w].copy()

                        delta_w = w_max - w
                        delta_h = h_max - h
                        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                        left, right = delta_w // 2, delta_w - (delta_w // 2)

                        obj_copy = obj.copy()
                        obj_copy.dtype = "uint8"

                        conv_hull = convex_hull_image(obj)
                        conv_hull_area = sum(sum(conv_hull))

                        label1_obj = I_label1_gray[y : y + h, x : x + w].copy()
                        label1_obj[obj_copy < 1] = 0

                        label2_obj = I_label2_gray[y : y + h, x : x + w].copy()
                        label2_obj[obj_copy < 1] = 0

                        obj_copy = cv2.copyMakeBorder(
                            obj_copy,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color_black,
                        )

                        obj.dtype = "uint8"
                        obj = obj * 255

                        area_list.append(area)
                        conv_hull_area_list.append(conv_hull_area)

                        label1_pixel_list.append(np.sum(label1_obj) / area)
                        label2_pixel_list.append(np.sum(label2_obj) / area)

                        check_obj_id = label_obj.copy()
                        check_obj_id[Mask_orig < 255] = 0
                        check_obj_id.dtype = "uint8"

                        if check_obj_id.max() > 0:
                            obj_id_list.append(1)
                        else:
                            obj_id_list.append(0)

                for k in range(len(area_list)):
                    writer.writerow(
                        {
                            "area": area_list[k] / sum(areas),
                            "conv_hull_area": conv_hull_area_list[k] / sum(areas),
                            "average_label1_pixel": label1_pixel_list[k] / 255,
                            "average_label2_pixel": label2_pixel_list[k] / 255,
                            "obj_id": obj_id_list[k],
                        }
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for the classification network"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="preactivation_resunet",
        help="choose model for inference",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="datasets/raw_chromosome_data",
        help="dataset folder with chromosome images",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=Dataset.img_size,
        help="input image size (default: 480x640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--weight-num",
        type=int,
        default=0,
        help="weight number for inference",
    )
    parser.add_argument(
        "--weight-path", type=str, default="", help="path to weight file"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="datasets/binary_classification_data/",
        help="directory of training data",
    )
    args = parser.parse_args()

    args.model = "resunet"
    args.weight_path = "resunet-20200423T1726"
    args.weight_num = 6

    os.makedirs(args.save_dir, exist_ok=True)

    inference(args)
    extract_features_and_save(args)
