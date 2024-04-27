import math
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import sigmoid_beta_schedule
import networkx as nx
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, PartialMarkovKernel
from .generative import *
from .inference import PpcGraphicalModel, asvi, mlp_amortizer

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
        pz_what = self.digit_features(K=self._num_digits, batch_shape=(B,))
        z_what = pyro.sample("z_what", pz_what)
        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            pz_where = self.digit_positions(z_where, K=self._num_digits,
                                            batch_shape=(B,))
            z_where = pyro.sample("z_where__%d" % t, pz_where)
            px = self.decoder(z_what, z_where)
            pyro.sample("X__%d" % t, px, obs=x)

    def guide(self, xs):
        B, T, _, _ = xs.shape

        data = xs.reshape(xs.shape[0], math.prod(xs.shape[1:]))
        with asvi(amortizer=self.encoders, data=data, event_shape=xs.shape[1:],
                  namer=lambda n: n.split('__')[0]):
            pz_what = self.digit_features(self._num_digits, batch_shape=(B,))
            pyro.sample("z_what", pz_what)

        z_where = None
        for t, x in pyro.markov(enumerate(xs.unbind(1))):
            x = x.reshape(x.shape[0], math.prod(x.shape[1:]))
            with asvi(amortizer=self.encoders, data=x, event_shape=x.shape[1:],
                      namer=lambda n: n.split('__')[0]):
                pz_where = self.digit_positions(z_where, K=self._num_digits,
                                                batch_shape=(B,))
                z_where = pyro.sample('z_where__%d' % t, pz_where)

class MnistPpc(BaseModel):
    def __init__(self, digit_side=28, z_dims=[20, 128, 256]):
        super().__init__()
        self.prior = GaussianPrior(z_dims[0])
        self.decoder1 = ConditionalGaussian(z_dims[0], z_dims[1])
        self.decoder2 = ConditionalGaussian(z_dims[1], z_dims[2])
        self.likelihood = MlpBernoulliLikelihood(z_dims[2],
                                                 (digit_side, digit_side))

        self.graph = PpcGraphicalModel()
        self.graph.add_node("z1", [], self.prior)
        self.graph.add_node("z2", ["z1"], self.decoder1)
        self.graph.add_node("z3", ["z2"], self.decoder2)
        self.graph.add_node("X", ["z3"], self.likelihood)

    def forward(self, xs=None, **kwargs):
        if xs is not None:
            B = xs.shape[0]
            self.graph.clamp("X", xs)
        else:
            B = 1
        self.prior.batch_shape = (B,)
        self.decoder1.batch_shape = self.decoder2.batch_shape = (B,)
        self.likelihood.batch_shape = (B,)
        with clamp_graph(self.graph, X=xs) as graph:
            return graph.forward()

    def guide(self, xs=None, lr=1e-3):
        if xs is not None:
            B = xs.shape[0]
            self.graph.clamp("X", xs)
        else:
            B = 1
        self.prior.batch_shape = (B,)
        self.decoder1.batch_shape = self.decoder2.batch_shape = (B,)
        self.likelihood.batch_shape = (B,)
        with clamp_graph(self.graph, X=xs) as graph:
            return graph.guide(lr=lr)

class BouncingMnistPpc(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, num_digits=3, T=10,
                 x_side=96, z_what_dim=10, z_where_dim=2):
        super().__init__()
        self._num_digits = num_digits
        self._num_times = T

        self.decoder = DigitsDecoder(digit_side, hidden_dim, x_side, z_what_dim)
        self.digit_features = DigitFeatures(num_digits, z_what_dim)
        self.digit_positions = DigitPositions(num_digits, z_where_dim)

        self.graph = PpcGraphicalModel()
        self.graph.add_node("z_what", [], self.digit_features)
        for t in range(T):
            if t == 0:
                where_kernel = PartialMarkovKernel(self.digit_positions, None)
                self.graph.add_node("z_where__0", [], where_kernel)
            else:
                self.graph.add_node("z_where__%d" % t, ["z_where__%d" % (t-1)],
                                    self.digit_positions)
            self.graph.add_node("X__%d" % t, ["z_what", "z_where__%d" % t],
                                self.decoder)

    def forward(self, xs=None, **kwargs):
        B, T, _, _ = xs.shape if xs is not None else (1, self._num_times, 0, 0)
        self.digit_features.batch_shape = (B,)
        self.digit_positions.batch_shape = (B,)
        clamps = {"X__%d" % t: xs[:, t] for t in range(T) if xs is not None}
        with clamp_graph(self.graph, **clamps) as graph:
            recons = graph.forward()
        return torch.stack(recons, dim=2)

    def guide(self, xs=None, lr=1e-3):
        B, T, _, _ = xs.shape if xs is not None else (1, self._num_times, 0, 0)
        self.digit_features.batch_shape = (B,)
        self.digit_positions.batch_shape = (B,)
        clamps = {"X__%d" % t: xs[:, t] for t in range(T) if xs is not None}
        with clamp_graph(self.graph, **clamps) as graph:
            recons = graph.guide(lr=lr)
        return torch.stack(recons, dim=2)

class DiffusionPpc(BaseModel):
    def __init__(self, channels=3, dim_mults=(1, 2, 4, 8), thick=True,
                 flash_attn=True, hidden_dim=64, img_side=128, T=100):
        super().__init__()
        self._channels = channels
        self._img_side = img_side
        self._num_times = T

        self.diffusion = DiffusionStep(sigmoid_beta_schedule(T),
                                       dim_mults=dim_mults,
                                       flash_attn=flash_attn,
                                       hidden_dim=hidden_dim,
                                       thick=thick, x_side=img_side)
        self.prior = DiffusionPrior(channels, img_side)

        self.graph = PpcGraphicalModel()
        self.graph.add_node("X__%d" % T, [], self.prior)
        for t in reversed(range(T)):
            step_kernel = PartialMarkovKernel(self.diffusion, t=t)
            self.graph.add_node("X__%d" % t, ["X__%d" % (t+1)], step_kernel)

    def forward(self, xs=None, **kwargs):
        B, C, _, _ = xs.shape if xs is not None else (1, self._channels, 0, 0)
        self.diffusion.batch_shape = (B,)
        self.prior.batch_shape = (B,)
        with clamp_graph(self.graph, X__0=xs) as graph:
            return graph.forward()

    def guide(self, xs=None, lr=1e-4):
        B, C, _, _ = xs.shape if xs is not None else (1, self._channels, 0, 0)
        self.diffusion.batch_shape = (B,)
        self.prior.batch_shape = (B,)
        with clamp_graph(self.graph, X__0=xs) as graph:
            return graph.guide(lr=lr)
