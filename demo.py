import os
import sys
from collections import defaultdict
import argparse
import json
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torchcontrib.optim import SWA
import shap


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc1(x)


def f_function(x):
    a = x[:, 0]
    b = x[:, 1]
    c = x[:, 2]
    # return 3*a+2*b+10*a*b+c
    return 1 * a + 4 * b + c


if __name__ == "__main__":
    model = Net()
    ori_state_dict = model.state_dict()
    ori_state_dict["fc1.weight"].copy_(torch.Tensor([1, 4, 1]))
    ori_state_dict["fc1.bias"].copy_(torch.Tensor([0]))
    model.load_state_dict(ori_state_dict)
    x1 = torch.Tensor([1, 1, 2]).reshape(1, -1)
    x2 = torch.Tensor([1, 2, 3]).reshape(1, -1)
    x_zero = torch.zeros_like(x1)
    y1 = f_function(x1)

    e = shap.DeepExplainer(model, x_zero.reshape(1, -1))
    shap_values = e.shap_values(x1.reshape(1, -1))
    print(
        "pred",
        model(x1.reshape(1, -1)).data,
        "baseline",
        model(x_zero.reshape(1, -1)).data,
        "label",
        y1,
        "shap",
        shap_values,
    )
