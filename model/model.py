import math
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .inference import asvi, mlp_amortizer

class DigitPositions(BaseModel):
    def __init__(self, z_where_dim=2):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_where_dim))
        self.register_buffer('scale', torch.ones(z_where_dim) * 0.2)

    def forward(self, z_where, t=0, K=3, batch_shape=()):
        scale = self.scale
        if z_where is None:
            scale = scale * 5
        prior = dist.Normal(self.loc, scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return pyro.sample("z_where__%d" % t, prior.to_event(2))

class DigitFeatures(BaseModel):
    def __init__(self, z_what_dim=10):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_what_dim))
        self.register_buffer('scale', torch.ones(z_what_dim))

    def forward(self, K=3, batch_shape=()):
        prior = dist.Normal(self.loc, self.scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return pyro.sample("z_what", prior.to_event(2))

class DigitsDecoder(BaseModel):
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

    def forward(self, what, where, t=0, x=None):
        P, B, K, _ = where.shape
        digits = self.decoder(what)
        frame = torch.clamp(self.blit(digits, where).sum(-3), 0., 1.)
        likelihood = dist.ContinuousBernoulli(frame).to_event(2)
        return pyro.sample("X__%d" % t, likelihood, obs=x)

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
            z_where = self.digit_positions(t, K=self._num_digits,
                                           batch_shape=(B,), z_where=z_where)
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
                z_where = self.digit_positions(t, self._num_digits,
                                               batch_shape=(B,),
                                               z_where=z_where)
