import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import yaml
import inspect
import sys

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

models = {
    "unet": "U-Net",
    "unet": "U-Net",
    "resunet": "Residual U-Net",
    "nested_unet": "U-Net++",
    "attention_unet": "Attention U-Net",
    "cenet": "CE-Net",
    "segnet": "SegNet",
    "fcn_resnet101": "FCN8s",
    "deeplabv3_resnet101": "DeepLab v3",
    "pspnet": "PSPNet",
    "proposed_cnn": "Proposed CNN",
}

others = {
    "adaptive_thresholding": "Adaptive Thresholding",
    "histogram_analysis": "Histogram Analysis",
}

others2 = {
    "proposed_cnn_bcn": "Proposed CNN + BCN",
}

best_all = glob.glob("./**/best_all.yaml", recursive=True)
eval_others = glob.glob("./**/eval_others.yaml", recursive=True)

with open(best_all[0]) as f:
    best_threshold_metrics = yaml.load(f, Loader=yaml.FullLoader)

with open(eval_others[0]) as f:
    eval_others_metrics = yaml.load(f, Loader=yaml.FullLoader)

print("\n")
print(
    "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
        "Model", "Dice (%)", "Se (%)", "Sp (%)", "Pre (%)", "Acc (%)"
    )
)  #'Ch1 DSC',

for _, other in enumerate(others):
    for _, t_val in enumerate(eval_others_metrics[other]):
        m = f"{others[other]}"
        dsc = eval_others_metrics[other]["DSC"]
        se = eval_others_metrics[other]["SE"]
        sp = eval_others_metrics[other]["SP"]
        pre = eval_others_metrics[other]["PRE"]
        acc = eval_others_metrics[other]["ACC"]

    print(
        "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
            m, dsc, se, sp, pre, acc
        )
    )

for _, model in enumerate(models):
    for _, t_val in enumerate(best_threshold_metrics[model]):
        m = f"{models[model]} (t={t_val})"
        dsc = best_threshold_metrics[model][t_val]["DSC"]
        se = best_threshold_metrics[model][t_val]["SE"]
        sp = best_threshold_metrics[model][t_val]["SP"]
        pre = best_threshold_metrics[model][t_val]["PRE"]
        acc = best_threshold_metrics[model][t_val]["ACC"]

    print(
        "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
            m, dsc, se, sp, pre, acc
        )
    )

for _, other in enumerate(others2):
    for _, t_val in enumerate(eval_others_metrics[other]):
        m = f"{others2[other]}"
        dsc = eval_others_metrics[other]["DSC"]
        se = eval_others_metrics[other]["SE"]
        sp = eval_others_metrics[other]["SP"]
        pre = eval_others_metrics[other]["PRE"]
        acc = eval_others_metrics[other]["ACC"]

    print(
        "{:<25}  {:<10}  {:<10}  {:<10}  {:<10}  {:<10}".format(
            m, dsc, se, sp, pre, acc
        )
    )
