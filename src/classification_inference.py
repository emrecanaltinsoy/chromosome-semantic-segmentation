import os
import sys
import inspect
import argparse

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.Classification_model import classification_model
from classification_dataset import ChromoNonChromoDataset as Dataset


def main(args):
    # args.weights_path = "classification_model-20210401T1305"
    # args.weight_num = 41

    if not args.weights_path:
        print("Choose weights path")
        sys.exit()
    if not args.weight_num:
        print("Choose a weight number")
        sys.exit()

    weight_path = "output/{}/{}/weights".format(Dataset.name, args.weights_path)

    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    model = classification_model(
        Dataset.in_channels, Dataset.out_channels, [200, 100, 50, 25, 5]
    )
    model.to(device)

    try:
        model_name = glob(weight_path + "/epoch-{}*".format(args.weight_num))[0]
        state_dict = torch.load(model_name, map_location=device)
    except IndexError:
        print("Weight file is not found!")
        sys.exit()

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = []

    loader_set = data_loaders(args)
    loaders = {"test": loader_set}

    correct = 0
    total = 0
    test_img_num = 0
    true_chromosome = 0
    false_chromosome = 0
    true_nonchromosome = 0
    false_nonchromosome = 0

    for _, datum in enumerate(loaders["test"], 0):
        data, y_true = datum
        data, y_true = data.to(device, dtype=torch.float), y_true.to(
            device, dtype=torch.float
        )

        with torch.set_grad_enabled(False):
            y_pred = model(data)
            y_true = y_true.long()

            sm = nn.Softmax(dim=1)
            pred_percentage = sm(y_pred)

            _, preds = torch.max(pred_percentage, 1)
            total += y_true.size(0)
            correct += (preds == y_true).sum().item()

            loss = criterion(y_pred, y_true)

            preds_np = preds.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()

            for img_num in range(preds_np.shape[0]):
                if preds_np[img_num]:
                    if y_true_np[img_num]:
                        true_chromosome += 1
                    else:
                        false_chromosome += 1
                elif y_true_np[img_num]:
                    false_nonchromosome += 1
                else:
                    true_nonchromosome += 1
                test_img_num += 1

            total_loss.append(loss.item())

    print(f"mean loss={np.mean(total_loss)}")
    print(f"Accuracy = {(100 * correct / total)} %%")
    print(
        f"TC = {true_chromosome}, FC = {false_chromosome}, TNC = {true_nonchromosome}, FNC = {false_nonchromosome}"
    )


def data_loaders(args):
    dataset_test = datasets(args)
    return DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
    )


def datasets(args):
    return Dataset(file_dir=args.test_data, subset="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation of G-banding chromosome Images"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="input batch size for training (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
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
        "--weights-path", type=str, default=None, help="path where the weights are"
    )
    parser.add_argument("--weight-num", type=str, default=None, help="weight number")
    parser.add_argument(
        "--test-data",
        type=str,
        default="./datasets/binary_classification_data/test_data.csv",
        help="directory of validation data",
    )
    args = parser.parse_args()
    main(args)
