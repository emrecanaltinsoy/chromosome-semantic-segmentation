import os
import inspect
import sys

import cv2
import numpy as np
import skimage.io as io
import yaml


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def dsc_score(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    TP = np.logical_and(im1, im2).sum()
    TN = np.logical_and((1.0 - im1), (1.0 - im2)).sum()
    FP = np.logical_and(im1, (1.0 - im2)).sum()
    FN = np.logical_and((1.0 - im1), im2).sum()

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    dsc_ = (2.0 * TP) / (FP + FN + (2 * TP))

    return [dsc_, TP, TN, FP, FN]


def eval_func(im1, im2):
    dsc_, TP, TN, FP, FN = dsc_score(im1, im2)
    se_ = TP / (TP + FN)
    sp_ = TN / (TN + FP)
    pre_ = TP / (TP + FP)
    acc_ = (TP + TN) / (TP + TN + FP + FN)

    return [dsc_, se_, sp_, pre_, acc_, TP, TN, FP, FN]


def find_threshold_dsc(models, y_preds, y_trues, t_vals, print_vals=False, prefix=""):
    if print_vals:
        print("\n")
        print("=" * 54)
        print(prefix)
        print("=" * 54)
        print("{:<20}  {:<20}  {:<10}".format("model name", "Best DSC", "Best T_val"))
        print("-" * 54)

    evals_dict_all = {}
    t_vals_dict = {}
    for model in models:
        evals_dict = {}
        t_max = 0
        dsc_max = 0
        for t_val in t_vals[model]:
            evals = []
            for im_num in range(84):
                y_pred = y_preds[model][im_num]
                y_true = y_trues[im_num]

                _, y_pred_bin = cv2.threshold(y_pred, t_val, 255, cv2.THRESH_BINARY)

                evals.append(dsc_score(y_true, y_pred_bin))

            dsc, TP, TN, FP, FN = [float(x) for x in np.mean(evals, axis=0)]
            evals_dict[t_val] = {
                "DSC": round(dsc * 100, 4),
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
            }

            if dsc > dsc_max:
                dsc_max = dsc
                t_max = t_val

        evals_dict_all[model] = evals_dict
        t_vals_dict[model] = t_max

        os.makedirs(f"logs/thresh_eval/{model}", exist_ok=True)
        with open(f"logs/thresh_eval/{model}/{model}_{prefix}.yaml", "w") as fp:
            yaml.dump(evals_dict, fp)

        if print_vals:
            print("{:<20}  {:<20}  {:<10}".format(model, dsc_max, t_max))
            print("-" * 54)

    return t_vals_dict


def save_best_metrics_images(models, y_preds, y_trues, t_vals, prefix=""):
    evals_dict_all = {}
    for model in models:
        evals_dict = {}
        evals = []
        for im_num in range(84):
            y_pred = y_preds[model][im_num]
            y_true = y_trues[im_num]

            _, y_pred_bin = cv2.threshold(y_pred, t_vals[model], 255, cv2.THRESH_BINARY)

            """ 
            final_img = im_origs[im_num].copy()
            final_img[y_pred_bin==0]=255
            final_img.dtype = 'uint8'
            io.imsave(os.path.join(pred_path,"{}_final.png".format(im_num)), final_img)  
            """

            os.makedirs(f"output/results/{model}", exist_ok=True)
            io.imsave(f"output/results/{model}/{im_num}_{model}.png", y_pred_bin)

            evals.append(eval_func(y_true, y_pred_bin))

        dsc, se, sp, pre, acc, TP, TN, FP, FN = [
            float(x) for x in np.mean(evals, axis=0)
        ]
        evals_dict[t_vals[model]] = {
            "DSC": round(dsc * 100, 4),
            "SE": round(se * 100, 4),
            "SP": round(sp * 100, 4),
            "PRE": round(pre * 100, 4),
            "ACC": round(acc * 100, 4),
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        }

        evals_dict_all[model] = evals_dict
        print(f"{model} threshold images are saved")

    with open(f"logs/thresh_eval/{prefix}_all.yaml", "w") as fp:
        yaml.dump(evals_dict_all, fp)


if __name__ == "__main__":
    orig_path = "./datasets/raw_chromosome_label1"
    y_trues = [
        cv2.imread(os.path.join(orig_path, "{}_label1_orig.png".format(im_num)))[
            :, :, 0
        ]
        for im_num in range(84)
    ]

    models = [
        "unet",
        "resunet",
        "proposed_cnn",
        "cenet",
        "segnet",
        "nested_unet",
        "attention_unet",
        "fcn_resnet101",
        "deeplabv3_resnet101",
        "pspnet",
    ]
    #'unet', 'resunet', 'proposed_cnn', 'cenet', 'segnet', 'nested_unet', 'attention_unet', 'fcn_resnet101', 'deeplabv3_resnet101', 'pspnet'

    y_preds = {}
    for model in models:
        pred_path = f"./logs/scalar/{model}/test"
        y_pred = [
            cv2.imread(os.path.join(pred_path, "{}_label1.png".format(im_num + 1)))[
                :, :, 0
            ]
            for im_num in range(84)
        ]

        y_preds[model] = y_pred

    t_step = 5
    t_vals = {}
    for m in models:
        t_vals[m] = list(range(5, 255, t_step))
        t_vals[m].append(254)
        t_vals[m].append(1)
        t_vals[m] = sorted(t_vals[m])
    t_vals_search = find_threshold_dsc(
        models, y_preds, y_trues, t_vals, print_vals=True, prefix="grid_search"
    )

    t_vals = {}
    for m in models:
        t_vals[m] = list(
            range(t_vals_search[m] - t_step + 1, t_vals_search[m] + t_step)
        )
    t_vals_optimum = find_threshold_dsc(
        models, y_preds, y_trues, t_vals, print_vals=True, prefix="search_best"
    )

    save_best_metrics_images(models, y_preds, y_trues, t_vals_optimum, prefix="best")
