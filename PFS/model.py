import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_size):
        super(Regressor, self).__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.layers(x)


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
