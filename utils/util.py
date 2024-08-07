from collections import OrderedDict
from denoising_diffusion_pytorch.learned_gaussian_diffusion import discretized_gaussian_log_likelihood
from itertools import repeat
import json
import math
import pandas as pd
from pathlib import Path
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

from .thirdparty import soft_clamp

class DiscretizedGaussian(dist.TorchDistribution):
    def __init__(self, loc, scale, min=-1, max=1, bins=256):
        self._bin_width = (max - min) / bins
        self._bins = bins
        self._max, self._min = max, min
        self._normal = dist.Normal(loc, scale)

    @property
    def batch_shape(self):
        return self._normal.batch_shape

    @property
    def event_shape(self):
        return self._normal.event_shape

    @property
    def has_rsample(self):
        return True

    @property
    def loc(self):
        return self._normal.loc

    def log_prob(self, value):
        if len(value.shape) < len(self.loc.shape):
            value = value.expand(*self.loc.shape)
        return discretized_gaussian_log_likelihood(value, means=self.loc,
                                                   log_scales=self.scale.log())

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    def rsample(self, sample_shape=torch.Size([])):
        samples = self._normal.rsample(sample_shape=sample_shape)
        samples_frac = samples.frac()
        samples = samples - samples_frac
        samples = samples + (samples_frac / self._bin_width).round() *\
                  self._bin_width
        return soft_clamp(samples, self.min, self.max)

    @property
    def support(self):
        return torch.distributions.constraints.interval(self.min, self.max)

    @property
    def scale(self):
        return self._normal.scale

class Conv2dUpsampleBlock(nn.Module):
    def __init__(self, in_chans, in_side, out_chans, out_side, kernel_size=4,
                 stride=2, padding=1, nonlinearity=nn.SiLU):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, 1, 1),
            nn.BatchNorm2d(in_chans, track_running_stats=False),
            nonlinearity()
        )
        # in_chans x in_side x in_side -> out_chans x out_side x out_side
        self.upsample = nn.Sequential(
            nn.Upsample(size=(out_side, out_side)),
            nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chans, track_running_stats=False),
            nonlinearity()
        )

    def forward(self, features):
        return self.upsample(self.residual(features) + features)

class ConvTransposeBlock2d(nn.Module):
    def __init__(self, in_chans, in_side, out_chans, out_side,
                 nonlinearity=nn.SiLU):
        super().__init__()

        # in_chans x in_side x in_side -> in_chans x in_side x in_side
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 4, 2, 1),
            nn.BatchNorm2d(out_chans, track_running_stats=False),
            nonlinearity(),
            nn.ConvTranspose2d(out_chans, out_chans, 4, 2, 1),
            nn.BatchNorm2d(out_chans, track_running_stats=False),
            nonlinearity(),
        )
        # out_chans x in_side x in_side -> out_chans x out_side x out_side
        padding = deconv2d_padding(in_side, out_side, 4, 2)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_chans+out_chans, out_chans, 4, 2, padding[0]),
            nn.BatchNorm2d(out_chans, track_running_stats=False),
            nonlinearity(),
        )

    def forward(self, features):
        hidden = torch.cat((features, self.bottleneck(features)), dim=-3)
        return self.upsample(hidden)

def deconv2d_padding(input_dim, output_dim, kernel_size, stride):
    """
    Calculate the padding and output_padding required for a ConvTranspose layer to achieve the desired output dimension.

    Parameters:
    input_dim (int): The size of the input dimension (height or width).
    output_dim (int): The desired size of the output dimension (height or width).
    stride (int): The stride of the convolution.
    kernel_size (int): The size of the convolution kernel.

    Returns:
    tuple: The required padding and output_padding to achieve the desired output dimension.
    """
    no_padding_output_dim = (input_dim - 1) * stride + kernel_size

    padding = math.ceil(max(no_padding_output_dim - output_dim, 0) / 2)
    output_padding = max(output_dim - (no_padding_output_dim - 2 * padding), 0)

    assert no_padding_output_dim - 2 * padding + output_padding == output_dim

    return padding, output_padding

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
