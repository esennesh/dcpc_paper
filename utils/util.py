from collections import OrderedDict
from itertools import repeat
import json
import math
import pandas as pd
from pathlib import Path
import pyro
import torch

class ScoreNetwork0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self, x_side=28):
        super().__init__()
        self._x_side = x_side
        nch = 3
        chs = [32, 64, 128, 256, 256]
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
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3], chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2], chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1], chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], x_side, x_side)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0], chs[0], kernel_size=3, padding=1),  # (batch, chs[0], x_side, x_side)
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

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = tconv(signals[-i] + signal)
        signal = signal.reshape(*x2.shape)  # (..., 1 * 28 * 28)
        return signal


def log_likelihood(trace):
    return sum(site['log_prob'] for site in trace.nodes.values()
               if site['type'] == 'sample' and site['is_observed'])

def log_joint(trace):
    return sum(site['log_prob'] for site in trace.nodes.values()
               if site['type'] == 'sample')

def logmeanexp(logits, dim=0, keepdim=True):
    return torch.logsumexp(logits, dim, keepdim) - math.log(logits.shape[dim])

def importance(model, guide, *args, **kwargs):
    tq = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
    with pyro.poutine.replay(trace=tq):
        tp = pyro.poutine.trace(model).get_trace(*args, **kwargs)
    tq.compute_log_prob()
    tp.compute_log_prob()
    return tp, log_joint(tp) - log_joint(tq)

def regen_trace(model, trace, *args, **kwargs):
    with pyro.poutine.replay(trace=trace):
        trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
    trace.compute_log_prob()
    return trace

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
