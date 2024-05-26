from collections import OrderedDict
from itertools import repeat
import json
import math
import pandas as pd
from pathlib import Path
import pyro
import torch

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
