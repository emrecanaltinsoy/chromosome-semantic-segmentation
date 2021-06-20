import os
import random
import datetime
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import pandas as pd

from utils import adjustData


class ChromoNonChromoDataset(Dataset):

    in_channels = 4
    out_channels = 2
    now = datetime.datetime.now()
    name = "binary_classification"

    def __init__(
        self,
        file_dir,
        subset="train",
        random_sampling=True,
        seed=42,
    ):
        assert subset in ["train", "validation", "test"]

        self.subset = subset

        print("reading {} data...".format(subset))

        df = pd.read_csv(file_dir)
        areas = list(df["area"])
        conv_hull_area = list(df["conv_hull_area"])
        average_label1_pixel = list(df["average_label1_pixel"])
        average_label2_pixel = list(df["average_label2_pixel"])
        classes = list(df["obj_id"])

        areas_npy = np.array(areas)
        conv_hull_area_npy = np.array(conv_hull_area)
        average_label1_pixel_npy = np.array(average_label1_pixel)
        average_label2_pixel_npy = np.array(average_label2_pixel)
        classes_npy = np.array(classes)

        if self.subset != "test":
            indices = np.arange(classes_npy.shape[0])
            np.random.seed(seed)
            np.random.shuffle(indices)
            areas_npy = areas_npy[indices]
            conv_hull_area_npy = conv_hull_area_npy[indices]
            average_label1_pixel_npy = average_label1_pixel_npy[indices]
            average_label2_pixel_npy = average_label2_pixel_npy[indices]
            classes_npy = classes_npy[indices]

        data = np.stack(
            [
                areas_npy,
                conv_hull_area_npy,
                average_label1_pixel_npy,
                average_label2_pixel_npy,
            ],
            axis=1,
        )

        self.len = len(classes_npy)

        self.classes_tensor = torch.from_numpy(classes_npy)
        self.data_tensor = torch.from_numpy(data)

        print("done creating {} dataset".format(subset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.classes_tensor[idx]
