import math
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, MarkovKernel
from .inference import asvi, mlp_amortizer, PpcGraph

class DigitPositions(MarkovKernel):
    def __init__(self, z_where_dim=2):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_where_dim))
        self.register_buffer('scale', torch.ones(z_where_dim) * 0.2)

    def forward(self, z_where, K=3, batch_shape=()) -> dist.Distribution:
        scale = self.scale
        if z_where is None:
            scale = scale * 5
        prior = dist.Normal(self.loc, scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return prior.to_event(2)

class DigitFeatures(MarkovKernel):
    def __init__(self, z_what_dim=10):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_what_dim))
        self.register_buffer('scale', torch.ones(z_what_dim))

    def forward(self, K=3, batch_shape=()) -> dist.Distribution:
        prior = dist.Normal(self.loc, self.scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return prior.to_event(2)

class DigitsDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, x_side=96, z_what_dim=10):
        super().__init__()
        self._digit_side = digit_side
        self._x_side = x_side
        self.decoder = nn.Sequential(
            nn.Linear(z_what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )
        scale = torch.diagflat(torch.ones(2) * x_side / digit_side)
        self.register_buffer('scale', scale)
        self.translate = (x_side - digit_side) / digit_side

    def blit(self, digits, z_where):
        P, B, K, _ = z_where.shape
        affine_p1 = self.scale.repeat(P, B, K, 1, 1)
        affine_p2 = z_where.unsqueeze(-1) * self.translate
        affine_p2[:, :, :, 0, :] = -affine_p2[:, :, :, 0, :]
        grid = F.affine_grid(
            torch.cat((affine_p1, affine_p2), -1).view(P*B*K, 2, 3),
            torch.Size((P*B*K, 1, self._x_side, self._x_side)),
            align_corners=True
        )

        digits = digits.view(P*B*K, self._digit_side, self._digit_side)
        frames = F.grid_sample(digits.unsqueeze(1), grid, mode='nearest',
                               align_corners=True).squeeze(1)
        return frames.view(P, B, K, self._x_side, self._x_side)

    def forward(self, what, where) -> dist.Distribution:
        P, B, K, _ = where.shape
        digits = self.decoder(what)
        frame = torch.clamp(self.blit(digits, where).sum(-3), 0., 1.)
        return dist.ContinuousBernoulli(frame).to_event(2)

class DigitDecoder(BaseModel):
    def __init__(self, digit_side=28, hidden_dim=400, z_dim=10):
        super().__init__()
        self._digit_side = 28
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )

    def forward(self, what, x=None):
        P, B, _, _ = what.shape
        estimate = self.decoder(what).view(P, B, 1, self._digit_side,
                                           self._digit_side)
        likelihood = dist.ContinuousBernoulli(estimate).to_event(3)
        return pyro.sample("X", likelihood, obs=x)

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
