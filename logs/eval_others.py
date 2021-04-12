import numpy as np
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import inspect
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def eval_func(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    TP = np.logical_and(im1, im2)
    TN = np.logical_and((1.-im1),  (1.-im2))
    FP = np.logical_and(im1, (1.-im2))
    FN = np.logical_and((1.-im1), im2)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    dsc_ = (2. * TP.sum()) / (FP.sum() + FN.sum() + (2 * TP.sum()))
    se_ = TP.sum() / (TP.sum() + FN.sum())
    sp_ = TN.sum() / (TN.sum() + FP.sum())
    pre_ = TP.sum() / (TP.sum() + FP.sum())
    acc_ = (TP.sum() + TN.sum()) / (TP.sum() + TN.sum() + FP.sum() + FN.sum())

    return [dsc_, se_, sp_, pre_, acc_, TP.sum(), TN.sum(), FP.sum(), FN.sum()]

if __name__ == "__main__":
    folder_name = ['adaptive_thresholding', 'histogram_analysis', 'proposed_cnn_bcn']
    img_pref = ['adaptive_gaussian', 'chromosome', 'proposed_cnn_bnn']
    zipped = zip(folder_name, img_pref)

    label1_folder_path = 'output/results/label1_origs/'
    label1_orig_pref = 'label1_orig'

    evals_dict_all = {}
    for method, pref in zipped:
        folder_path = f'output/results/{method}/'
        evals_dict = {}
        evals = list()
        for im_num in range(84):
            label1_orig = imread(os.path.join(label1_folder_path,"{}_{}.png".format(im_num,label1_orig_pref)), as_gray=True)
            
            im_res = cv2.imread(os.path.join(folder_path,"{}_{}.png".format(im_num,pref)))[:,:,0]

            evals.append(eval_func(label1_orig, im_res))

        dsc, se, sp, pre, acc, TP, TN, FP, FN = [float(x) for x in np.mean(evals, axis=0)]
        evals_dict = {
            'DSC': round(dsc*100, 4),
            'SE': round(se*100, 4),
            'SP': round(sp*100, 4),
            'PRE': round(pre*100, 4),
            'ACC': round(acc*100, 4),
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
        }

        evals_dict_all[method] = evals_dict
        print(f'{method} images checked')

    with open(f'logs/thresh_eval/eval_others.yaml', "w") as fp:
        yaml.dump(evals_dict_all, fp)
