import argparse
import yaml
import os
import numpy as np
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pytorch_model_summary import summary

from logger import Logger
from models.Classification_model import classification_model
from binary_classification_dataset import ChromoNonChromoDataset as Dataset

def main(args):    
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    model = classification_model(Dataset.in_channels, Dataset.out_channels, [200,100,50,25,5])
    print(summary(model, torch.zeros((args.batch_size, Dataset.in_channels)), show_input=False, show_hierarchical=False))
    model.to(device)

    if args.weights == "":
        args.weights = "./output/{}/{}-{:%Y%m%dT%H%M}/weights".format(Dataset.name, model.net_name, Dataset.now)
    if args.logs == "":
        args.logs = "./output/{}/{}-{:%Y%m%dT%H%M}/logs".format(Dataset.name, model.net_name, Dataset.now)

    make_dirs(args)
    save_args(args)
    best_validation_loss = 1.0
    step = 0
    train_step = 0
    val_step = 0
    best_epoch = 0
    epoch_step = 0

    loss_func_name = 'CrossEntropyLoss'
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr = args.lr)  # Stochastic gradient descent
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []
    acc_train = []
    acc_valid = []

    phases = ["train", "valid"]
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    for epoch in range(1, args.epochs+1):
        correct = 0
        total = 0
        for phase in phases:
            if phase == "train":
                model.train()
            elif phase == "valid":
                model.eval()

            for i, datum in enumerate(loaders[phase], 0):
                
                data, y_true = datum

                if phase == "train":
                    step += 1
                    epoch_step += 1
                    
                data, y_true = data.to(device, dtype=torch.float), y_true.to(device, dtype=torch.float)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(data)
                    sm = nn.Softmax(dim=1)
                    pred_percentage = sm(y_pred)
                    y_true = y_true.long()
                    
                    _, preds = torch.max(pred_percentage, 1)
                    total += y_true.size(0)
                    correct += (preds == y_true).sum().item()

                    loss = criterion(y_pred, y_true)

                    if phase == "valid":
                        val_step += 1
                        loss_valid.append(loss.item())

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()

                        optimizer.step()
                        if (step) % 100 == 0:
                            train_step += 1
                            print('Epoch={}/{}, step={}, loss={}'.format(epoch,args.epochs,epoch_step,loss.item()))

            if phase == "train":
                mean_train_loss = np.mean(loss_train)
                log_loss_summary(logger, mean_train_loss, epoch, prefix="loss")
                acc_train = 100 * correct / total
                log_loss_summary(logger, acc_train, epoch, prefix="acc")
                correct = 0
                total = 0
                loss_train = []

            if phase == "valid":
                validation_loss = np.mean(loss_valid)
                log_loss_summary(logger, validation_loss, epoch, prefix="val_loss")
                acc_valid = 100 * correct / total
                log_loss_summary(logger, acc_valid, epoch, prefix="val_acc")
                correct = 0
                total = 0
                loss_valid = []

                if validation_loss < best_validation_loss:
                    print('saving weights...')
                    best_epoch = epoch
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), os.path.join(args.weights, "epoch-{}-val_loss-{}-val_acc-{}-{}.pt".format(best_epoch,best_validation_loss,acc_valid,loss_func_name))) 
                
            scheduler.step()
            epoch_step = 0
        print('Epoch={}/{}, loss={}, val_loss={}, acc={}, val_acc={}'.format(epoch,args.epochs,
                    mean_train_loss, 
                    validation_loss,
                    acc_train,
                    acc_valid))

def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
    )
    return loader_train, loader_valid

def datasets(args):
    train = Dataset(
        file_dir=args.train_data,
        subset="train",
    )
    valid = Dataset(
        file_dir=args.validation_data,
        subset="validation",
    )
    return train, valid

def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix, np.mean(loss), step)

def make_dirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

def save_args(args):
    args_file = os.path.join(args.logs, "args.yaml")
    with open(args_file, "w") as fp:
        yaml.dump(vars(args), fp)

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
        "--weights", 
        type=str, 
        default="", 
        help="folder to save weights"
    )
    parser.add_argument(
        "--logs", 
        type=str, 
        default="", 
        help="folder to save logs"
    )
    parser.add_argument(
        "--train-data", 
        type=str, 
        default="./datasets/binary_classification_data/train_data.csv", 
        help="directory of training data"
    )
    parser.add_argument(
        "--validation-data", 
        type=str, 
        default="./datasets/binary_classification_data/valid_data.csv", 
        help="directory of validation data"
    )
    args = parser.parse_args()
    main(args)
