import math
import networkx as nx
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .generative import *
from .inference import asvi, mlp_amortizer, PpcGraph

class BouncingMnistAsvi(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, num_digits=3, T=10,
                 x_side=96, z_what_dim=10, z_where_dim=2):
        super().__init__()
        self._num_digits = num_digits

        self.encoders = pnn.PyroModule[nn.ModuleDict]({
            'mean_fields': pnn.PyroModule[nn.ModuleDict]({
                'z_what': pnn.PyroModule[nn.ModuleDict]({
                    'loc': mlp_amortizer(T * x_side ** 2, hidden_dim,
                                         num_digits * z_what_dim),
                    'scale': mlp_amortizer(T * x_side ** 2, hidden_dim,
                                           num_digits * z_what_dim),
                }),
                'z_where': pnn.PyroModule[nn.ModuleDict]({
                    'loc': mlp_amortizer(x_side ** 2, hidden_dim,
                                         num_digits * z_where_dim),
                    'scale': mlp_amortizer(x_side ** 2, hidden_dim,
                                           num_digits * z_where_dim),
                })
            }),
            'mixing_logits': pnn.PyroModule[nn.ModuleDict]({
                'z_what':  mlp_amortizer(T * x_side ** 2,
                                         num_digits * z_what_dim, 1),
                'z_where':  mlp_amortizer(x_side ** 2, num_digits * z_where_dim,
                                          1)
            }),
        })

        self.decoder = DigitsDecoder(digit_side, hidden_dim, x_side, z_what_dim)
        self.digit_features = DigitFeatures(z_what_dim)
        self.digit_positions = DigitPositions(z_where_dim)

    def forward(self, xs):
        return self.model(xs)

    def model(self, xs):
        B, T, _, _ = xs.shape
        z_what = self.digit_features(K=self._num_digits, batch_shape=(B,))
        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            z_where = self.digit_positions(z_where, t=t, K=self._num_digits,
                                           batch_shape=(B,))
            self.decoder(t, z_what, z_where, x)

    def guide(self, xs):
        B, T, _, _ = xs.shape

        data = xs.reshape(xs.shape[0], math.prod(xs.shape[1:]))
        with asvi(amortizer=self.encoders, data=data, event_shape=xs.shape[1:],
                  namer=lambda n: n.split('__')[0]):
            self.digit_features(self._num_digits, batch_shape=(B,))

        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            x = x.reshape(x.shape[0], math.prod(x.shape[1:]))
            with asvi(amortizer=self.encoders, data=x, event_shape=x.shape[1:],
                      namer=lambda n: n.split('__')[0]):
                z_where = self.digit_positions(z_where, t=t, K=self._num_digits,
                                               batch_shape=(B,))

class MnistPpc(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, temperature=1e-3,
                 z_dim=10):
        super().__init__()
        self.digit_features = DigitFeatures(z_dim)
        self.decoder = DigitDecoder(digit_side, hidden_dim, z_dim)

        self.graph = PpcGraph(temperature)
        self.graph.add_node("z_what", [], self.digit_features)
        self.graph.add_node("X", ["z_what"], self.decoder)

    def forward(self, xs=None):
        if xs is not None:
            B, _, _, _ = xs.shape
        else:
            B = 1

        self.graph.set_kwargs("z_what", K=1, batch_shape=(B,))
        z = self.digit_features(K=1, batch_shape=(B,))
        self.graph.set_kwargs("X", x=xs)
        return self.decoder(z, x=xs)

class BouncingMnistPpc(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, num_digits=3, T=10,
                 temperature=1e-3, x_side=96, z_what_dim=10, z_where_dim=2):
        super().__init__()
        self._num_digits = num_digits

        self.decoder = DigitsDecoder(digit_side, hidden_dim, x_side, z_what_dim)
        self.digit_features = DigitFeatures(z_what_dim)
        self.digit_positions = DigitPositions(z_where_dim)

        self.graph = PpcGraph(temperature)
        self.graph.add_node("z_what", [], self.digit_features)
        for t in range(T):
            if t == 0:
                where_kernel = lambda **kwargs: self.digit_positions(None,
                                                                     **kwargs)
                self.graph.add_node("z_where__0", [], where_kernel)
            else:
                self.graph.add_node("z_where__%d" % t, ["z_where__%d" % (t-1)],
                                    self.digit_positions)
            self.graph.add_node("X__%d" % t, ["z_what", "z_where__%d" % t],
                                self.decoder)

    def forward(self, xs):
        B, T, _, _ = xs.shape
        self.graph.set_kwargs("z_what", K=self._num_digits, batch_shape=(B,))
        z_what = self.digit_features(K=self._num_digits, batch_shape=(B,))
        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            self.graph.set_kwargs("z_where__%d" % t, t=t, K=self._num_digits,
                                  batch_shape=(B,))
            z_where = self.digit_positions(z_where, t=t, K=self._num_digits,
                                           batch_shape=(B,))

            self.graph.set_kwargs("X__%d" % t, t=t, x=x)
            self.decoder(z_what, z_where, t=t, x=x)
