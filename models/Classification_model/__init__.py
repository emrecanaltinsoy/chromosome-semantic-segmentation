import torch
import torch.nn as nn


class classification_model(nn.Module):
    def __init__(self, n_in, n_out, layers):
        super().__init__()

        all_layers = []
        self.n_in = n_in
        self.n_out = n_out

        for i in layers:
            all_layers.append(nn.Linear(self.n_in, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            self.n_in = i

        all_layers.append(nn.Linear(layers[-1], self.n_out))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    net_name = "classification_model"
