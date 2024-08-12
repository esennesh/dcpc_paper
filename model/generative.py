from abc import abstractmethod
from contextlib import contextmanager
from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.simple_diffusion import UViT
import functools
import math
import networkx as nx
import numpy as np
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
from torch.distributions import biject_to
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, ImportanceModel, MarkovKernel
from base import MarkovKernelApplication
from utils.util import DiscretizedGaussian
from utils.thirdparty import NLVM, ScoreNetwork0, soft_clamp

class DigitPositions(MarkovKernel):
    def __init__(self, hidden_dim=10, num_digits=3, z_where_dim=2):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_where_dim))
        self.register_buffer('scale', torch.ones(z_where_dim) * 0.2)
        self.dynamics = nn.Sequential(
            nn.Linear(z_where_dim * num_digits, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_where_dim * num_digits)
        )
        self.batch_shape = ()
        self._num_digits = num_digits

    @property
    def event_dim(self):
        return 2

    def forward(self, z_where, obs=None) -> dist.Distribution:
        param_shape = (*self.batch_shape, self._num_digits, *self.loc.shape)
        scale = self.scale.expand(param_shape)
        if z_where is None:
            loc = self.loc.expand(param_shape)
            scale = scale * 5
        else:
            P, B, K, D = z_where.shape
            loc = self.dynamics(z_where.view(P, B, K * D)).view(P, B, K, D)
        return dist.Normal(loc, scale).to_event(2)

class DigitFeatures(MarkovKernel):
    def __init__(self, num_digits=3, z_what_dim=10):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_what_dim))
        self.register_buffer('scale', torch.ones(z_what_dim))
        self.batch_shape = ()
        self._num_digits = num_digits

    @property
    def event_dim(self):
        return 2

    def forward(self, obs=None) -> dist.Distribution:
        dist_shape = (*self.batch_shape, self._num_digits, *self.loc.shape)
        return dist.Normal(self.loc, self.scale).expand(dist_shape).to_event(2)

class DigitsDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, x_side=96, z_what_dim=10,
                 mnist_mean=None):
        super().__init__()
        self.batch_shape = ()
        self._digit_side = digit_side
        self._x_side = x_side
        self.decoder = nn.Sequential(
            nn.Linear(z_what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )
        self.register_buffer('scale', torch.eye(2) * x_side / digit_side)

        self.translate = (x_side - digit_side) / digit_side
        self._digits = {}

    def blit(self, digits, z_where):
        P, B, K, _ = z_where.shape
        affine_p1 = self.scale.repeat(P, B, K, 1, 1)
        affine_p2 = soft_clamp(z_where.unsqueeze(-1) * self.translate, -1, 1)
        affine_p2[:, :, :, 0, :] = -affine_p2[:, :, :, 0, :]
        grid = F.affine_grid(
            torch.cat((affine_p1, affine_p2), -1).view(P*B*K, 2, 3),
            torch.Size((P*B*K, 1, self._x_side, self._x_side)),
            align_corners=False
        )

        digits = digits.view(P*B*K, self._digit_side, self._digit_side)
        digits = digits.transpose(-1, -2).unsqueeze(1)
        frames = F.grid_sample(digits, grid, mode='nearest',
                               align_corners=False).squeeze(1).transpose(-1, -2)
        return frames.view(P, B, K, self._x_side, self._x_side)

    @property
    def event_dim(self):
        return 2

    def forward(self, what, where, obs=None) -> dist.Distribution:
        P, B, K, _ = where.shape
        if what not in self._digits:
            self._digits = {
                what: self.decoder(what)
            }
        digits = self._digits[what]
        frames = soft_clamp(self.blit(digits, where).sum(dim=-3), 0., 1.)
        return dist.ContinuousBernoulli(frames).to_event(2)

class DigitDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, z_dim=10):
        super().__init__()
        self.batch_shape = ()
        self._digit_side = 28
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )

    @property
    def event_dim(self):
        return 3

    def forward(self, what, obs=None) -> dist.Distribution:
        P, B, _, _ = what.shape
        estimate = self.decoder(what).view(P, B, 1, self._digit_side,
                                           self._digit_side)
        return dist.ContinuousBernoulli(estimate).to_event(3)

class GaussianPrior(MarkovKernel):
    def __init__(self, out_dim, train_params=True):
        super().__init__()
        self.batch_shape = ()

        if train_params:
            self.loc = nn.Parameter(torch.zeros(out_dim))
            self.covariance = nn.Parameter(torch.eye(out_dim))
        else:
            self.register_buffer("loc", torch.zeros(out_dim))
            self.register_buffer("covariance", torch.eye(out_dim))

    @property
    def event_dim(self):
        return 1

    def forward(self, obs=None) -> dist.Distribution:
        loc = self.loc.expand(*self.batch_shape, *self.loc.shape)
        scale = torch.tril(self.covariance).expand(*self.batch_shape,
                                                   *self.covariance.shape)
        return dist.MultivariateNormal(loc, scale_tril=scale)

class ConditionalGaussian(MarkovKernel):
    def __init__(self, in_dim, out_dim, nonlinearity=nn.ReLU):
        super().__init__()
        self.batch_shape = ()

        self.covariance = nn.Parameter(torch.eye(out_dim))
        self.decoder = nn.Sequential(
            nonlinearity(), nn.Linear(in_dim, out_dim),
        )

    @property
    def event_dim(self):
        return 1

    def forward(self, hs: torch.Tensor, obs=None) -> dist.Distribution:
        scale = torch.tril(self.covariance).expand(*self.batch_shape,
                                                   *self.covariance.shape)
        return dist.MultivariateNormal(self.decoder(hs), scale_tril=scale)

class GaussianSsm(MarkovKernel):
    def __init__(self, z_dim, u_dim=0, nonlinearity=nn.Identity):
        super().__init__()
        self.batch_shape = ()

        self._u_dim = u_dim
        if u_dim:
            self.control_dynamics = nn.Sequential(
                nonlinearity(), nn.Linear(u_dim, z_dim, bias=False)
            )
        self.covariance = nn.Parameter(torch.eye(z_dim))
        self.state_dynamics = nn.Sequential(
            nonlinearity(), nn.Linear(z_dim, z_dim, bias=False)
        )

    @property
    def event_dim(self):
        return 1

    def forward(self, z: torch.Tensor, u=None, obs=None) -> dist.Distribution:
        assert (self._u_dim > 0) == (u is not None)

        scale = torch.tril(self.covariance).expand(*self.batch_shape,
                                                   *self.covariance.shape)
        z_next = self.state_dynamics(z)
        if self._u_dim:
            z_next = z_next + self.control_dynamics(u)
        return dist.MultivariateNormal(z_next, scale_tril=scale)

class GaussianEmission(MarkovKernel):
    def __init__(self, z_dim, x_dim, u_dim=0, nonlinearity=nn.Identity):
        super().__init__()
        self.batch_shape = ()

        self._u_dim = u_dim
        if u_dim:
            self.control_emission = nn.Sequential(
                nonlinearity(), nn.Linear(u_dim, x_dim, bias=False)
            )
        self.covariance = nn.Parameter(torch.eye(x_dim))
        self.emission = nn.Sequential(
            nonlinearity(), nn.Linear(z_dim, x_dim, bias=False)
        )

    @property
    def event_dim(self):
        return 1

    def forward(self, z: torch.Tensor, u=None, obs=None) -> dist.Distribution:
        assert (self._u_dim > 0) == (u is not None)

        scale = torch.tril(self.covariance).expand(*self.batch_shape,
                                                   *self.covariance.shape)
        x = self.emission(z)
        if self._u_dim:
            x = x + self.control_emission(u)
        return dist.MultivariateNormal(x, scale_tril=scale)

class MlpBernoulliLikelihood(MarkovKernel):
    def __init__(self, in_dim, out_shape, nonlinearity=nn.ReLU):
        super().__init__()
        self.batch_shape = ()
        self._out_shape = out_shape

        self.decoder = nn.Sequential(
            nonlinearity(), nn.Linear(in_dim, math.prod(self._out_shape)),
        )

    @property
    def event_dim(self):
        return 1 + len(self._out_shape)

    def forward(self, hs: torch.Tensor, obs=None) -> dist.Distribution:
        P, B, _ = hs.shape
        logits = self.decoder(hs).view(P, B, 1, *self._out_shape)
        return dist.ContinuousBernoulli(logits=logits).to_event(self.event_dim)

class DiffusionPrior(MarkovKernel):
    def __init__(self, channels=3, img_side=128):
        super().__init__()
        self.batch_shape = ()
        self.register_buffer('loc', torch.zeros(channels, img_side, img_side))
        self.register_buffer('scale', torch.ones(channels, img_side, img_side))

    @property
    def event_dim(self):
        return 3

    def forward(self, obs=None) -> dist.Distribution:
        loc = self.loc.expand(*self.batch_shape, *self.loc.shape)
        scale = self.scale.expand(*self.batch_shape, *self.scale.shape)
        return dist.Normal(loc, scale).to_event(3)

class DiffusionStep(MarkovKernel):
    def __init__(self, betas, x_side=128, unet="Unet", dim_mults=(1, 2, 4, 8),
                 flash_attn=True, hidden_dim=64):
        super().__init__()
        self.batch_shape = ()
        self.register_buffer('betas', betas.to(dtype=torch.float))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

        if unet == "Unet":
            self.unet = Unet(dim=hidden_dim, dim_mults=dim_mults,
                             flash_attn=flash_attn)
        elif unet == "UViT":
            self.unet = UViT(x_side, out_dim=3, channels=3,
                             dim_mults=dim_mults)
        else:
            self.unet = ScoreNetwork0(x_side)

    @property
    def event_dim(self):
        return 3

    def forward(self, xs_prev: torch.Tensor, t=0, obs=None):
        P, B, C, W, H = xs_prev.shape
        score = self.unet(xs_prev.view(P*B, C, W, H),
                          torch.tensor(t, device=xs_prev.device,
                                       dtype=torch.long).repeat(P*B))
        score = score.view(*xs_prev.shape)
        beta = self.betas[t]
        alpha, alpha_bar = self.alphas[t], self.alpha_bars[t]
        loc = 1/alpha.sqrt() * (xs_prev -
                                (beta / (1. - alpha_bar).sqrt()) * score)
        return dist.Normal(loc, beta).to_event(3)

class ConvolutionalEncoder(pnn.PyroModule):
    def __init__(self, channels=3, z_dim=40, hidden_dim=256, img_side=64):
        super().__init__()
        self._channels = channels
        self._hidden_dim = hidden_dim
        self._img_side = img_side
        self._z_dim = z_dim

        self.convs = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1), # 3 x 64 x 64 -> 32 x 32 x 32
            nn.BatchNorm2d(32, track_running_stats=False), nn.SiLU(),
            nn.Conv2d(32, 32, 4, 2, 1), # 32 x 32 x 32 -> 32 x 16 x 16
            nn.BatchNorm2d(32, track_running_stats=False), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 32 x 16 x 16 -> 64 x 8 x 8
            nn.BatchNorm2d(64, track_running_stats=False), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), # 64 x 8 x 8 -> 64 x 4 x 4
            nn.BatchNorm2d(64, track_running_stats=False), nn.SiLU(),
            nn.Conv2d(64, hidden_dim, 4, 1, 0), # 64 x 4 x 4 -> 256 x 1 x 1
            nn.BatchNorm2d(hidden_dim, track_running_stats=False), nn.SiLU(),
        )
        self.linear = nn.Linear(hidden_dim, z_dim * 2)

    def forward(self, xs: torch.Tensor) -> dist.Distribution:
        B, _, _, _ = xs.shape
        hs = self.linear(self.convs(xs).squeeze()).view(B, self._z_dim, 2)
        loc, log_scale = hs.unbind(dim=-1)
        return dist.Normal(loc, log_scale.exp() + 1e-5)

class ConvolutionalDecoder(MarkovKernel):
    def __init__(self, channels=3, z_dim=40, img_side=64, nonlinearity=nn.Tanh,
                 discretize=True, hidden_dim=256):
        super().__init__()
        self.batch_shape = ()
        self._channels = channels
        self._discretize = discretize
        self._hidden_dim = hidden_dim
        self._img_side = img_side

        self.linear = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.SiLU()
        )
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 4, 1, 0), # 256 x 1 x 1 -> 64 x 4 x 4
            nn.BatchNorm2d(64, track_running_stats=False), nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # 64 x 4 x 4 -> 64 x 8 x 8
            nn.BatchNorm2d(64, track_running_stats=False), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 64 x 8 x 8 -> 32 x 16 x 16
            nn.BatchNorm2d(32, track_running_stats=False), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # 32 x 16 x 16 -> 32 x 32 x 32
            nn.BatchNorm2d(32, track_running_stats=False), nn.SiLU(),
            # 32 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nonlinearity()
        )

    @property
    def event_dim(self):
        return 3

    def forward(self, zs: torch.Tensor, obs=None) -> dist.Distribution:
        P, B, _ = zs.shape
        hs = self.linear(zs.view(P*B, -1))
        hs = hs.view(P*B, self._hidden_dim, 1, 1)
        hs = self.convs(hs).view(P, B, self._channels, self._img_side,
                                 self._img_side)
        if self._discretize:
            return DiscretizedGaussian(hs, 1e-2).to_event(3)
        return dist.Normal(hs, 1e-2).to_event(3)

class FixedVarianceDecoder(MarkovKernel):
    def __init__(self, channels=3, img_side=64, scale=0.01, z_dim=64):
        super().__init__()
        self.batch_shape = ()
        self._channels = channels
        self._img_side = img_side

        self.likelihood_scale = scale
        self.mean_network = NLVM(z_dim, channels, nonlinearity=F.tanh)

    @property
    def event_dim(self):
        return 3

    def forward(self, zs: torch.Tensor, obs=None) -> dist.Distribution:
        P, B, _ = zs.shape
        loc = self.mean_network(zs.view(P*B, -1)).view(P, B, self._channels,
                                                       self._img_side,
                                                       self._img_side)
        return dist.Normal(loc, self.likelihood_scale).to_event(3)

class GraphicalModel(ImportanceModel, pnn.PyroModule):
    def __init__(self):
        super().__init__()
        self._graph = nx.DiGraph()

    def add_node(self, site, parents, kernel):
        self._graph.add_node(site, is_observed=False, kernel=kernel, kwargs={},
                             support=None, value=None)
        for parent in parents:
            self._graph.add_edge(parent, site)

    def child_sites(self, site):
        return self._graph.successors(site)

    def clamp(self, site, value):
        assert value is not None
        self.nodes[site]['is_observed'] = True
        return self.update(site, value)

    def clear(self):
        for site in self.nodes:
            self.unclamp(site)

    @contextmanager
    def condition(self, **kwargs):
        kwargs = {k: v for (k, v) in kwargs.items() if torch.is_tensor(v)}
        try:
            for k, v in kwargs.items():
                self.clamp(k, v)
            yield self
        finally:
            for k in kwargs:
                self.unclamp(k)

    @abstractmethod
    def conditioner(self, data):
        raise NotImplementedError

    def model(self, **kwargs):
        results = ()

        for site, kernel in self.sweep():
            obs = self.nodes[site]['value'] if self.nodes[site]['is_observed']\
                  else None
            density = kernel(*self.parent_vals(site), **{"obs": obs})
            self.nodes[site]['support'] = density.support
            self.update(site, pyro.sample(site, density, obs=obs).detach())

            if len(list(self.child_sites(site))) == 0:
                results = results + (self.nodes[site]['value'],)
        return results[0] if len(results) == 1 else results

    def kernel(self, site):
        apply = self.nodes[site]['kernel']
        return functools.partial(getattr(self, apply.kernel), *apply.args,
                                 **apply.kwargs)

    def log_prob(self, site, value, *args, **kwargs):
        return self.kernel(site)(*args, **kwargs).log_prob(value)

    @property
    def nodes(self):
        return self._graph.nodes

    def parent_sites(self, site):
        return self._graph.predecessors(site)

    def parent_vals(self, site):
        return tuple(self.nodes[p]['value'] for p in self.parent_sites(site))

    def predict(self, *args, B=1, P=1, **kwargs):
        with self.condition(**kwargs) as conditioned:
            return conditioned.forward(*args, B=B, mode="prior", P=P)

    @functools.cached_property
    def stochastic_nodes(self):
        return [site for site in self.nodes
                if not self.nodes[site]['is_observed']]

    def sweep(self, forward=True, observations=True):
        for site in self.topological_sort(not forward):
            if self.nodes[site]["is_observed"] and not observations:
                continue
            yield site, self.kernel(site)

    @functools.cache
    def topological_sort(self, reverse=False):
        nodes = list(nx.lexicographical_topological_sort(self._graph))
        if reverse:
            nodes = list(reversed(nodes))
        return nodes

    def unclamp(self, site):
        for key in self.nodes[site]:
            if key != "kernel":
                self.nodes[site][key] = None

    def update(self, site, value):
        self.nodes[site]['value'] = value
        return self.nodes[site]['value']
