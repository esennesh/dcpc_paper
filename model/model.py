import math
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import sigmoid_beta_schedule
import networkx as nx
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, MarkovKernelApplication
from .generative import *
from .inference import PpcGraphicalModel, asvi, mlp_amortizer

class BouncingMnistAsvi(ImportanceModel):
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

    def generate(self, xs):
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

class MnistPpc(PpcGraphicalModel):
    def __init__(self, x_dims, z_dims=[20, 128, 256]):
        super().__init__()
        self.prior = GaussianPrior(z_dims[0])
        self.decoder1 = ConditionalGaussian(z_dims[0], z_dims[1])
        self.decoder2 = ConditionalGaussian(z_dims[1], z_dims[2])
        self.likelihood = MlpBernoulliLikelihood(z_dims[2],
                                                 (x_dims[-1], x_dims[-1]))

        self.add_node("z1", [], MarkovKernelApplication("prior", (), {}))
        self.add_node("z2", ["z1"], MarkovKernelApplication("decoder1", (),
                                                            {}))
        self.add_node("z3", ["z2"], MarkovKernelApplication("decoder2", (),
                                                            {}))
        self.add_node("X", ["z3"], MarkovKernelApplication("likelihood", (),
                                                           {}))

    def conditioner(self, data):
        return {"X": data}

class BouncingMnistPpc(PpcGraphicalModel):
    def __init__(self, dims, digit_side=28, hidden_dim=400, num_digits=3,
                 z_what_dim=10, z_where_dim=2, mnist_mean=None):
        super().__init__()
        self._num_digits = num_digits
        self._num_times = dims[0]

        self.decoder = DigitsDecoder(digit_side, hidden_dim, dims[-1],
                                     z_what_dim, mnist_mean)
        self.digit_features = DigitFeatures(num_digits, z_what_dim)
        self.digit_positions = DigitPositions(num_digits, z_where_dim)

        self.add_node("z_what", [],
                      MarkovKernelApplication("digit_features", (), {}))
        for t in range(dims[0]):
            if t == 0:
                where_kernel = MarkovKernelApplication("digit_positions",
                                                       (None,), {})
                self.add_node("z_where__0", [], where_kernel)
            else:
                self.add_node(
                    "z_where__%d" % t, ["z_where__%d" % (t-1)],
                    MarkovKernelApplication("digit_positions", (), {})
                )
            self.add_node("X__%d" % t, ["z_what", "z_where__%d" % t],
                          MarkovKernelApplication("decoder", (), {}))

    def conditioner(self, xs):
        T = xs.shape[1]
        return {"X__%d" % t: xs[:, t] for t in range(T)}

class DiffusionPpc(PpcGraphicalModel):
    def __init__(self, dims, dim_mults=(1, 2, 4, 8), unet="Unet",
                 flash_attn=True, hidden_dim=64, T=100):
        super().__init__()
        self._channels = dims[0]
        self._num_times = T

        self.diffusion = DiffusionStep(sigmoid_beta_schedule(T),
                                       dim_mults=dim_mults,
                                       flash_attn=flash_attn,
                                       hidden_dim=hidden_dim,
                                       unet=unet, x_side=dims[-1])
        self.prior = DiffusionPrior(dims[0], dims[-1])

        self.add_node("X__%d" % T, [], MarkovKernelApplication("prior", (),
                                                               {}))
        for t in reversed(range(T)):
            step_kernel = MarkovKernelApplication("diffusion", (), {"t": t})
            self.add_node("X__%d" % t, ["X__%d" % (t+1)], step_kernel)

    def forward(self, xs=None, **kwargs):
        B, C, _, _ = xs.shape if xs is not None else (1, self._channels, 0, 0)
        if B == 1 and 'B' in kwargs:
            B = kwargs.pop('B')
        self.diffusion.batch_shape = (B,)
        self.prior.batch_shape = (B,)
        return super().forward(X__0=xs, B=B, **kwargs)

class CelebAPpc(PpcGraphicalModel):
    def __init__(self, dims, z_dim=40, hidden_dim=256):
        super().__init__()
        self._channels = dims[0]

        self.prior = GaussianPrior(z_dim, False)
        self.likelihood = ConvolutionalDecoder(dims[0], z_dim, hidden_dim,
                                               dims[-1])

        self.add_node("z", [], MarkovKernelApplication("prior", (), {}))
        self.add_node("X", ["z"], MarkovKernelApplication("likelihood", (),
                                                          {}))

    def forward(self, xs=None, **kwargs):
        B, C, _, _ = xs.shape if xs is not None else (1, self._channels, 0, 0)
        if B == 1 and 'B' in kwargs:
            B = kwargs.pop('B')
        self.prior.batch_shape = (B,)
        self.likelihood.batch_shape = (B,)
        return super().forward(X=xs, B=B, **kwargs)
