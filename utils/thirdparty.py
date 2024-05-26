from collections import OrderedDict
from itertools import repeat
import json
import math
import pandas as pd
from pathlib import Path
import pyro
import torch
from torch import nn
import torch.nn.functional as F

class Deterministic(nn.Module):
    """
    The Deterministic Layer used in NLVM.
    """
    def __init__(self, in_dim: int, out_dim: int, activation=F.gelu):
        super(Deterministic, self).__init__()

        self.activation = activation

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1,
                              padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1,
                               padding=1)

        self.bn = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = out + x  # Skip connection
        return out

class Projection(nn.Module):
    """
    The Projection Layer used in NLVM.
    """
    def __init__(self, in_dim: int, ngf: int = 16, coef: int = 4,
                 activation=F.gelu):
        super(Projection, self).__init__()

        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(in_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5,
                                          stride=1, padding=2, bias=False)
        self.linear_bn = nn.BatchNorm1d(coef * ngf * ngf)
        self.deconv1_bn = nn.BatchNorm2d(ngf * coef)

    def forward(self, x):
        out = self.linear(x)
        out = self.linear_bn(out)
        out = self.activation(out)
        out = out.view(out.size(0), self.coef, self.ngf, self.ngf).contiguous()
        out = self.deconv1(out)
        out = self.deconv1_bn(out)
        out = self.activation(out)
        return out

class Output(nn.Module):
    """
    The Output Layer used in NLVM.
    """
    def __init__(self, x_in: int, nc: int, nonlinearity=F.tanh):
        super(Output, self).__init__()
        self.nonlinearity = nonlinearity
        self.output_layer = nn.ConvTranspose2d(x_in, nc, kernel_size=4,
                                               stride=2, padding=1)

    def forward(self, x: TensorType[..., 'n_channels', 'in_dim1', 'in_dim2']
                ) -> TensorType[...,  'n_channels', 'out_dim1', 'out_dim2']:
        out = self.output_layer(x)
        out = self.nonlinearity(out)
        return out

class NLVM(nn.Module):
    """
    Implementation of the model, taken from: https://github.com/juankuntz/ParEM/blob/main/torch/parem/models.py.
    Similar to https://github.com/enijkamp/short_run_inf.
    """
    def __init__(self, x_dim: int = 1, nc: int = 3, ngf: int = 16,
                 coef: int = 4, nonlinearity=F.tanh):
        super(NLVM, self).__init__()
        self.x_dim = x_dim
        self.ngf = ngf
        self.nc = nc

        self.projection_layer = Projection(x_dim, ngf=ngf, coef=coef)
        self.deterministic_layer_1 = Deterministic(ngf * coef, ngf * coef)
        self.deterministic_layer_2 = Deterministic(ngf * coef, ngf * coef)
        self.output_layer = Output(ngf * coef, nc, nonlinearity)

    def forward(self, x):
        out = self.projection_layer(x)
        out = self.deterministic_layer_1(out)
        out = self.deterministic_layer_2(out)
        out = self.output_layer(out)
        return out

class ScoreNetwork0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self, x_side=28):
        super().__init__()
        self._x_side = x_side
        nch = 3
        chs = [32, 64, 128, 256, 512]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(nch+1, chs[0], kernel_size=3, padding=1),  # (batch, ch, x_side, x_side)
                torch.nn.LogSigmoid(),  # (batch, 8, x_side, x_side)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(chs[4] * 16, chs[4] * 16), torch.nn.SiLU(),
            torch.nn.Linear(chs[4] * 16, chs[4] * 16)
        )
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], x_side, x_side)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], x_side, x_side)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], nch, kernel_size=3, padding=1),  # (batch, 1, x_side, x_side)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-3], 3, self._x_side, self._x_side))  # (..., ch0, x_side, x_side)
        tt = t[..., None, None, None].expand(t.shape[0], 1, self._x_side, self._x_side)  # (..., 3, x_side, x_side)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)
        signal = self.mlp(signal.view(signal.shape[0], -1)).view(*signal.shape)
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = signal.reshape(*x2.shape)  # (..., 1 * 28 * 28)
        return signal

def soft_clamp(z, low, high):
    z = torch.where(z > high, high - F.softplus(high - z), z)
    z = torch.where(z < low, low + F.softplus(z - low), z)
    return z
