import os
import inspect
import sys
import glob

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import yaml
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def main():
    grid_search = glob.glob(f"./**/*grid_search.yaml", recursive=True)
    short_range = glob.glob(f"./**/*search_best.yaml", recursive=True)
    for i in range(len(grid_search)):
        name = grid_search[i].split("\\")[-2]
        fig, axes = plt.subplots(1, 2)
        with open(grid_search[i]) as f:
            grid_scores = yaml.load(f, Loader=yaml.FullLoader)
        with open(short_range[i]) as f:
            short_range_scores = yaml.load(f, Loader=yaml.FullLoader)
        grid_x_axis = []
        grid_y_axis = []
        short_x_axis = []
        short_y_axis = []
        for _, key in enumerate(grid_scores):
            grid_x_axis.append(key)
            grid_y_axis.append(grid_scores[key]["DSC"])

        for _, key in enumerate(short_range_scores):
            short_x_axis.append(key)
            short_y_axis.append(short_range_scores[key]["DSC"])

        sns.lineplot(
            ax=axes[0],
            x=grid_x_axis,
            y=grid_y_axis,
            linewidth=5,
            linestyle="-",
            palette="flare",
        )
        sns.lineplot(
            ax=axes[1],
            x=short_x_axis,
            y=short_y_axis,
            linewidth=5,
            linestyle="-",
            palette="flare",
        )

        font_1 = 40
        font_2 = 30
        font_3 = 20
        fig.suptitle(f"{name}", fontsize=font_1)
        axes[0].set_title("Grid Search", fontsize=font_2)
        axes[0].set_xlabel("Threshold Values", fontsize=font_2)
        axes[0].set_ylabel("DSC Scores", fontsize=font_2)
        axes[0].set_xticks(np.arange(5, 256, step=font_2))
        axes[0].set_yticks(
            [np.min(grid_y_axis), np.median(grid_y_axis), np.max(grid_y_axis)]
        )
        axes[0].grid(linestyle="--", linewidth=0.5)
        axes[0].tick_params(axis="both", which="major", labelsize=font_3)
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        for tick in axes[0].get_xticklabels():
            tick.set_rotation(90)

        axes[1].set_title("Search Best", fontsize=font_2)
        axes[1].set_xlabel("Threshold Values", fontsize=font_2)
        axes[1].set_xticks(short_x_axis)
        axes[1].set_yticks(
            [np.min(short_y_axis), np.median(short_y_axis), np.max(short_y_axis)]
        )
        axes[1].grid(linestyle="--", linewidth=0.5)
        axes[1].tick_params(axis="both", which="major", labelsize=font_3)
        axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        for tick in axes[1].get_xticklabels():
            tick.set_rotation(90)

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()


if __name__ == "__main__":
    main()
