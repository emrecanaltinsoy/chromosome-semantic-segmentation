import os
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import random
import datetime
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset

from utils import adjustData

class RawChromosomeDataset(Dataset):

    in_channels = 1
    num_classes = 3
    now = datetime.datetime.now()
    name = 'raw_chromosome'
    img_size = [480, 640]

    def __init__(
        self,
        args,
        images_dir,
        image_size=img_size,
        subset="train",
        random_sampling=True,
        seed=42,
    ):
        assert subset in ["all", "train", "validation", "test", "test_diff"]

        self.subset = subset

        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))

        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".png" in f, filenames),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath, as_gray=True))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                patient_id = patient_id.split("\\")[-1]
                volumes[patient_id] = np.array(image_slices[0])
                masks[patient_id] = np.array(mask_slices[0])

        self.patients = sorted(volumes)

        test_cases =  int(len(volumes)/5)
        validation_cases =  int((len(volumes)-test_cases)/5)

        if subset != "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            self.patients = sorted(list(set(self.patients).difference(validation_patients)))
            test_patients = random.sample(self.patients, k=test_cases)
            if subset == "validation":
                self.patients = validation_patients
            elif subset == "test":
                self.patients = test_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(test_patients))
                )
                
        print("preprocessing {} volumes...".format(subset))
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("adjusting {} volumes...".format(subset))
        self.volumes = [adjustData(v, num_class=self.num_classes) for v in self.volumes]

        self.len = len(self.volumes)

        images_npy = [v[0] for v in self.volumes]
        images_npy = np.array(images_npy)
        masks_npy = [v[1] for v in self.volumes]
        masks_npy = np.array(masks_npy)

        self.random_sampling = random_sampling
        
        images_npy = images_npy.transpose(0, 3, 1, 2)
        masks_npy = masks_npy.transpose(0, 3, 1, 2)

        self.images_tensor = torch.from_numpy(images_npy)
        self.masks_tensor = torch.from_numpy(masks_npy)
        
        print("done creating {} dataset".format(subset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images_tensor[idx], self.masks_tensor[idx]
