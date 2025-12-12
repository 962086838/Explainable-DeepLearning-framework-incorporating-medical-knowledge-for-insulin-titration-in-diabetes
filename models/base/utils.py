from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.modules.pooling import _AvgPoolNd, _MaxPoolNd

from models.base.activations import SMU, SMU1


def get_activation(name: str = None) -> nn.Module:
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "prelu": nn.PReLU(),
        "selu": nn.SELU(inplace=True),
        "celu": nn.CELU(inplace=True),
        "glu": nn.GLU(),
        "elu": nn.ELU(inplace=True),
        "smu": SMU(),
        "smu1": SMU1(),
    }
    return activation_dict.get(name, nn.ReLU(inplace=True))


def get_norm(
    name: str, dim: int = 1, num_features: int = None, shape=None
) -> nn.Module:
    assert dim in (1, 2, 3), f"got a wrong dim for {name}"

    if name == "batch_norm":
        return getattr(nn, f"BatchNorm{dim}d")(num_features)
    elif name == "instance_norm":
        return getattr(nn, f"InstanceNorm{dim}d")(num_features)
    elif name == "layer_norm":
        return nn.LayerNorm(shape)


def get_rnn(name: str = "gru") -> Type[Union[nn.GRU, nn.LSTM]]:
    if name == "gru":
        return nn.GRU
    elif name == "lstm":
        return nn.LSTM


def get_pool(name: str = "max", dim: int = 1) -> Type[Union[_MaxPoolNd, _AvgPoolNd]]:
    assert dim in (1, 2, 3), f"got a wrong dim for {name}"

    if name == "max":
        return getattr(nn, f"MaxPool{dim}d")
    elif name == "avg":
        return getattr(nn, f"AvgPool{dim}d")


class InputLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        activation_name: str = "prelu",
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = get_activation(activation_name)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = self.act(out)
        out = self.dropout_1(out)
        return out


class EncoderRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_name: str = "gru",
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = get_rnn(rnn_name)(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            output, hidden = self.rnn(x)
        else:
            output, hidden = self.rnn(x, hidden)
        return output, hidden


class ResLinear(nn.Module):
    def __init__(
        self,
        features: int,
        num_block: int = 2,
        dropout: float = 0,
        activation_name: str = "selu",
    ):
        super().__init__()
        self.linear_list = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(num_block)]
        )

        self.act = get_activation(activation_name)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.linear_list:
            x = self.act(x + block(x))
            x = self.dropout_1(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        num_block: int = 2,
        dropout=0.3,
        activation_name: str = "selu",
    ):
        super().__init__()
        self.dropout = dropout
        self.fc1 = InputLinear(
            num_inputs, num_hidden, dropout=0, activation_name=activation_name
        )
        self.res_layer = ResLinear(num_hidden, num_block, dropout, activation_name)
        self.act = get_activation(activation_name)
        self.fc2 = InputLinear(
            num_hidden, num_outputs, dropout=0, activation_name=activation_name
        )
        self.norm = nn.LayerNorm(num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.res_layer(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class DenseSequential(nn.Module):
    def __init__(
        self,
        input_dims: int = 431,
        hidden_size: int = 128,
        activation_name: str = "relu",
        dropout: float = 0.5,
        output_dims: int = 1,
    ):
        super().__init__()
        self.fc_1 = nn.Linear(input_dims, hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.act = get_activation(activation_name)
        self.fc_2 = nn.Linear(hidden_size, output_dims)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc_1(x)
        out = self.dropout_1(out)
        out = self.act(out)
        out = self.fc_2(out)
        return out


class DenseSoftmaxSequential(nn.Module):
    def __init__(
        self,
        input_dims: int = 431,
        hidden_size: int = 128,
        activation_name: str = "relu",
        dropout: float = 0.5,
        output_dims: int = 1,
    ):
        super().__init__()
        self.fc_1 = nn.Linear(input_dims, hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.act = get_activation(activation_name)
        self.softmax = nn.Softmax()
        self.fc_2 = nn.Linear(hidden_size, output_dims)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc_1(x)
        out = self.dropout_1(out)
        out = self.act(out)
        out = self.fc_2(out)
        return out
