import torch
import torch.nn as nn
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super(MLP, self).__init__()
        self.twolayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.Dropout(0.4),       # 用于减少过拟合
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):

        y_pred = self.twolayernet(x)
        return y_pred