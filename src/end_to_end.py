import time
import torch
import torch.nn as nn
from glob import glob

import skimage.io as io
from skimage.io import imread
import cv2
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import numpy as np

import os
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.PreactivationResUNet import PreactResUNet
from models.Classification_model import classification_model


def feature_extraction(label1_dir, label2_dir):
    Im_label1 = cv2.imread(label1_dir)
    Im_label2 = cv2.imread(label2_dir)
    I_label1_gray = cv2.cvtColor(Im_label1, cv2.COLOR_BGR2GRAY)
    I_label2_gray = cv2.cvtColor(Im_label2, cv2.COLOR_BGR2GRAY)

    _, I_label1_bin = cv2.threshold(I_label1_gray, 0, 255, cv2.THRESH_BINARY)
    _, I_label2_bin = cv2.threshold(I_label2_gray, 255 * 0.99, 255, cv2.THRESH_BINARY)

    I_bin = I_label1_bin - I_label2_bin
    kernel = np.ones((3, 3), np.uint8)
    I_bin = cv2.morphologyEx(I_bin, cv2.MORPH_OPEN, kernel)

    _, I_bin = cv2.threshold(I_bin, 255 * 0.99, 255, cv2.THRESH_BINARY)

    connectivity = 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        I_bin, connectivity, cv2.CV_32S
    )

    area_list = []
    conv_hull_area_list = []
    label1_pixel_list = []
    label2_pixel_list = []
    label_num = []

    for num_labels_i in range(1, num_labels):
        label_obj = labels == num_labels_i

        x = stats[num_labels_i, cv2.CC_STAT_LEFT]
        y = stats[num_labels_i, cv2.CC_STAT_TOP]
        w = stats[num_labels_i, cv2.CC_STAT_WIDTH]
        h = stats[num_labels_i, cv2.CC_STAT_HEIGHT]
        area = stats[num_labels_i, cv2.CC_STAT_AREA]

        if area > 20:
            obj = label_obj[y : y + h, x : x + w].copy()

            obj_copy = obj.copy()
            obj_copy.dtype = "uint8"

            conv_hull = convex_hull_image(obj)
            conv_hull_area = sum(sum(conv_hull))

            label1_obj = I_label1_gray[y : y + h, x : x + w].copy()
            label1_obj[obj_copy < 1] = 0

            label2_obj = I_label2_gray[y : y + h, x : x + w].copy()
            label2_obj[obj_copy < 1] = 0

            label1_pixel = np.sum(label1_obj) / area
            label2_pixel = np.sum(label2_obj) / area

            area_list.append(area)
            conv_hull_area_list.append(conv_hull_area)
            label1_pixel_list.append(label1_pixel / 255)
            label2_pixel_list.append(label2_pixel / 255)
            label_num.append(num_labels_i)

    area_total = sum(area_list)
    area_list = area_list / area_total
    conv_hull_area_list = conv_hull_area_list / area_total

    areas_npy = np.array(area_list)
    conv_hull_area_npy = np.array(conv_hull_area_list)
    average_label1_pixel_npy = np.array(label1_pixel_list)
    average_label2_pixel_npy = np.array(label2_pixel_list)
    label_num_npy = np.array(label_num)

    all_data = np.stack(
        [
            areas_npy,
            conv_hull_area_npy,
            average_label1_pixel_npy,
            average_label2_pixel_npy,
        ],
        axis=1,
    )

    all_data = torch.from_numpy(all_data)
    label_num = torch.from_numpy(label_num_npy)

    return all_data, label_num


def final_res(label1_dir, label2_dir, final_labels, save_dir, img_orig):
    Im_label1 = cv2.imread(label1_dir)
    Im_label2 = cv2.imread(label2_dir)
    I_label1_gray = cv2.cvtColor(Im_label1, cv2.COLOR_BGR2GRAY)
    I_label2_gray = cv2.cvtColor(Im_label2, cv2.COLOR_BGR2GRAY)

    _, I_label1_bin = cv2.threshold(I_label1_gray, 0, 255, cv2.THRESH_BINARY)
    _, I_label2_bin = cv2.threshold(I_label2_gray, 255 * 0.99, 255, cv2.THRESH_BINARY)

    I_bin = I_label1_bin - I_label2_bin
    kernel = np.ones((3, 3), np.uint8)
    I_bin = cv2.morphologyEx(I_bin, cv2.MORPH_OPEN, kernel)

    _, I_bin = cv2.threshold(I_bin, 255 * 0.99, 255, cv2.THRESH_BINARY)

    connectivity = 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        I_bin, connectivity, cv2.CV_32S
    )

    chromo = np.zeros(I_bin.shape, np.uint8)
    non_chromo = np.zeros(I_bin.shape, np.uint8)

    for label in range(1, num_labels):
        obj = labels == label
        obj.dtype = "uint8"
        if label not in final_labels:
            chromo += obj
        else:
            non_chromo += obj

    # final_img = img_orig.copy()
    # final_img[chromo==0]=255
    # final_img.dtype = 'uint8'
    # io.imsave(os.path.join(save_dir,"{}_final.png".format(im_num)), final_img)

    chromo_res = os.path.join(save_dir, "{}_proposed_cnn_bcn.png".format(im_num))
    io.imsave(chromo_res, chromo * 255)


if __name__ == "__main__":
    test = "output/end_to_end/"
    save_dir = "output/end_to_end/"

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    segmentation_network = PreactResUNet(in_channels=1, num_classes=3, init_features=32)
    segmentation_network.to(device)

    weights = "logs/scalar/proposed_cnn/weights"
    seg_model_name = glob(weights + "/proposed_cnn-37*")[0]
    seg_state_dict = torch.load(seg_model_name, map_location=device)
    segmentation_network.load_state_dict(seg_state_dict)
    segmentation_network.eval()

    classification_network = classification_model(4, 2, [200, 100, 50, 25, 5])
    classification_network.to(device)

    weights = "output/binary_classification/classification_model-20210331T2310\weights"
    class_model_name = glob(weights + "/net-19*")[0]
    class_state_dict = torch.load(class_model_name, map_location=device)
    classification_network.load_state_dict(class_state_dict)
    classification_network.eval()

    start = time.time()
    print("clock started")
    for im_num in range(84):
        ################################################################################################
        ##############################           Segmentation        ###################################
        ################################################################################################
        filepath = test + "{}_image.png".format(im_num)
        img_orig = imread(filepath, as_gray=True)
        input_img = img_orig / 255

        new_input_img = np.zeros((1,) + input_img.shape + (1,))
        new_input_img[0, :, :, 0] = input_img
        input_img = new_input_img
        input_img = input_img.transpose(0, 3, 1, 2)

        input_img = torch.from_numpy(input_img)
        input_img = input_img.to(device, dtype=torch.float)

        y_pred = segmentation_network(input_img)

        ################################################################################################
        ############################         Feature Extraction        #################################
        ################################################################################################

        y_pred_np = y_pred.detach().cpu().numpy()

        label1_dir = os.path.join(test, "{}_label1.png".format(im_num))
        label2_dir = os.path.join(test, "{}_label2.png".format(im_num))
        io.imsave(label1_dir, y_pred_np[0, 1])
        io.imsave(label2_dir, y_pred_np[0, 2])

        features, labels = feature_extraction(label1_dir, label2_dir)

        ################################################################################################
        ############################           Classification        ###################################
        ################################################################################################

        features = features.to(device, dtype=torch.float)
        y_pred = classification_network(features)

        sm = nn.Softmax(dim=1)
        pred_percentage = sm(y_pred)
        final_percentage, preds = torch.max(pred_percentage, 1)

        final_labels = []
        labels = labels.detach().cpu().numpy()

        for i in range(len(preds)):
            if preds[i] == 0:
                final_labels.append(labels[i])

        final_res(label1_dir, label2_dir, final_labels, save_dir, img_orig)

    end = time.time()
    print("{} seconds past".format(end - start))
