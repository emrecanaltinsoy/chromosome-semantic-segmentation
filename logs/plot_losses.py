import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import yaml
import numpy as np
import argparse

files = glob.glob("./**/losses.yaml", recursive = True)

def main(args):
    #line_styles = ['dashdot', 'dotted', 'dashed', 'solid', 'dashdot', 'dotted', 'dashed', 'solid', 'dashdot', 'dotted']

    names =[]

    fig, ax = plt.subplots(1, 1)
    for i in range(len(files)):
        names.append(files[i].split('\\')[-2])
        with open(files[i]) as f:
            losses = yaml.load(f, Loader=yaml.FullLoader)
            sns.lineplot(data=losses[args.plot_loss], linewidth=5, linestyle='-', palette='flare')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)

    if args.plot_loss=='loss':
        plt.title('DSC Loss', fontsize=30)
    elif args.plot_loss=='val_loss':
        plt.title('Validation DSC Loss', fontsize=30)

    plt.xticks(np.arange(1, 41, step=3))
    plt.legend(names)
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.grid(linestyle='--', linewidth=0.5)

    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of G-banding chromosome Images"
    )
    parser.add_argument(
        "--plot-loss",
        type=str,
        default='val_loss',
        help="choose which values to print [loss, val_loss] (default: loss)",
    )
    args = parser.parse_args()
    main(args)
