import functools
import math
import networkx as nx
from typing import Callable, Sequence

import torch
import torch.distributions.constraints as constraints
from torch.distributions import biject_to, transform_to
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.distributions.distribution import Distribution
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr, helpful_support_errors, _product
from pyro.nn.module import PyroModule, PyroParam
from pyro.poutine.handlers import _make_handler
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.trace_struct import Trace
from pyro.poutine.util import site_is_subsample

from base import BaseModel
from .generative import GraphicalModel
import utils

def systematic_resample(log_weights):
    P, B = log_weights.shape

    positions = (torch.rand((B, P), device=log_weights.device) +\
                 torch.arange(P, device=log_weights.device).unsqueeze(0)) / P
    weights = F.softmax(log_weights, dim=0)
    cumsums = torch.cumsum(weights.transpose(0, 1), dim=1)
    (normalizers, _) = torch.max(input=cumsums, dim=1, keepdim=True)
    cumsums = cumsums / normalizers ## B * S

    index = torch.searchsorted(cumsums, positions).transpose(0, 1)
    assert index.shape == (P, B), "ERROR! systematic resampling resulted unexpected index shape."
    return torch.where(index >= P - 1, torch.zeros_like(index), index)

def _resample(log_weights, estimate_normalizer=False):
    indices = systematic_resample(log_weights)
    if estimate_normalizer:
        log_normalizer = utils.logmeanexp(log_weights)
        return indices, log_normalizer
    return indices

def _ancestor_index(indices, tensor):
    indices = indices.view(*indices.shape, *(1,) * len(tensor.shape[2:]))
    return torch.gather(tensor, 0, indices.expand(*tensor.shape))

class ParticleDict(nn.ParameterDict):
    def __init__(self, num_data, num_particles, batch_dim=1, particle_dim=0):
        super().__init__()
        self._batch_dim = batch_dim
        self._num_data = num_data
        self._num_particles = num_particles
        self._particle_dim = 0

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_particles(self):
        return self._num_particles

    def get_particles(self, key: str, idx: torch.LongTensor) -> torch.Tensor:
        val = self[key]
        assert val.shape[self._batch_dim] == self.num_data
        assert val.shape[self._particle_dim] == self.num_particles
        val = torch.index_select(val, self._batch_dim, idx.to(val.device))
        return val.to(idx.device)

    def set_particles(self, key: str, idx: torch.LongTensor, val: torch.Tensor):
        assert val.shape[self._particle_dim] == self.num_particles
        if key not in self:
            shape = list(val.shape)
            shape[self._batch_dim] = self.num_data
            self[key] = torch.zeros(*shape)
        with torch.no_grad():
            indices = idx.view((1,) * self._batch_dim + (len(idx),) +\
                               (1,) * len(val.shape[self._batch_dim+1:]))
            indices = indices.to(self[key].device)
            self[key].scatter_(self._batch_dim, indices.expand(val.shape),
                               val.to(self[key].device))

class DcpcGraphicalModel(GraphicalModel):
    def __init__(self, beta=0.99):
        super().__init__()
        self._beta = beta

    def _complete_conditional_error(self, site):
        error = self._site_errors(site)[0]
        for child in self.child_sites(site):
            site_index = list(self.parent_sites(child)).index(site)
            error = error + self._site_errors(child)[1 + site_index]
        return error

    def _compute_site_errors(self, site):
        value = self.nodes[site]['value']
        pvals = self.parent_vals(site)
        def logprobsum(value, *args, **kwargs):
            if self.nodes[site]['support']:
                value = biject_to(self.nodes[site]['support'])(value)
            return self.log_prob(site, value, *args, **kwargs).sum()

        if torch.is_floating_point(value):
            error = torch.func.grad(logprobsum,
                                    argnums=tuple(range(1+len(pvals))))
        else:
            raise NotImplementedError("Discrete prediction errors not implemented!")
        if self.nodes[site]['support']:
            value = biject_to(self.nodes[site]['support']).inv(value)
        return error(value, *pvals)

    def _site_errors(self, site):
        if self.nodes[site].get('errors', None) is None:
            self.nodes[site]['errors'] = self._compute_site_errors(site)
        return self.nodes[site]['errors']

    def add_node(self, site, parents, kernel):
        super().add_node(site, parents, kernel)
        self.nodes[site]['momentum'] = 0.

    @torch.no_grad()
    def get_posterior(self, name: str, event_dim: int, lr=1e-3):
        z = self.nodes[name]['value']
        if self.nodes[name]['support']:
            bijector = biject_to(self.nodes[name]['support'])
            z = bijector.inv(z)
        error = self._complete_conditional_error(name)
        fisher = error.var(dim=0, keepdim=True) + 1 / error.shape[0]
        prec = 1 / fisher
        prec = prec / ((1/prec.shape[-1]) * prec.sum(dim=-1, keepdim=True))

        proposal = dist.Normal(z + lr * prec * error, (2 * lr * prec).sqrt())
        proposal = proposal.to_event(event_dim)
        if self.nodes[name]['support']:
            proposal = dist.TransformedDistribution(proposal, [bijector])
        z_next = proposal.sample()

        log_cc = self.log_complete_conditional(name, z_next)
        log_proposal = proposal.log_prob(z_next)
        particle_indices, log_Zcc = _resample(log_cc - log_proposal,
                                              estimate_normalizer=True)
        z_next = _ancestor_index(particle_indices, z_next)
        log_cc = _ancestor_index(particle_indices, log_cc)
        return dist.Delta(z_next, log_cc - log_Zcc, event_dim=event_dim)

    def guide(self, lr=1e-3, **kwargs):
        results = ()
        for site, kernel in self.sweep(forward=False):
            if site in kwargs and kwargs[site] is not None:
                self.clamp(site, kwargs[site])
            if not self.nodes[site]["is_observed"]:
                posterior = self.get_posterior(site, kernel.func.event_dim,
                                               lr=lr)
                self.update(site, pyro.sample(site, posterior))

            if len(list(self.child_sites(site))) == 0:
                results = results + (self.nodes[site]['value'],)
        return results[0] if len(results) == 1 else results

    def log_complete_conditional(self, site, value):
        args = tuple(self.nodes[p]['value'] for p in self.parent_sites(site))
        log_sitecc = self.log_prob(site, value, *args)
        for child in self.child_sites(site):
            args = tuple(value if s == site else self.nodes[s]['value']
                         for s in self.parent_sites(child))
            log_sitecc = log_sitecc + self.log_prob(child,
                                                    self.nodes[child]['value'],
                                                    *args)
        return log_sitecc

    def update(self, site, value):
        self.nodes[site]['errors'] = None
        for child in self.child_sites(site):
            self.nodes[child]['errors'] = None
        return super().update(site, value)

def dist_params(dist: Distribution):
    return {k: v for k, v in dist.__dict__.items() if k[0] != '_'}

def mlp_amortizer(dom, hidden, cod):
    return nn.Sequential(
        nn.Linear(dom, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, hidden), nn.ReLU(),
        nn.Linear(hidden, cod)
    )

class AsviMessenger(TraceMessenger):
    def __init__(
        self,
        amortizer: PyroModule,
        data: torch.Tensor,
        event_shape: torch.Size,
        namer: Callable=None
    ):
        super().__init__()
        if not isinstance(amortizer, PyroModule):
            raise ValueError("Expected PyroModule for ASVI amortization")
        self._amortizer = (amortizer,)
        if not isinstance(data, torch.Tensor):
            raise ValueError("Expected tensor for ASVI conditioning")
        try:
            obs = data.view(-1, *event_shape)
        except:
            raise ValueError("Expected batch_shape + event_shape but found data shape %s" % data.shape)
        self.data = data
        self._event_shape = event_shape
        self._namer = namer

    @property
    def amortizer(self):
        return self._amortizer[0]

    def _get_params(self, name: str, prior: Distribution):
        if self._namer is not None:
            name = self._namer(name)
        batch_shape = self.data.shape[:-len(self._event_shape)]
        mixing_logits = deep_getattr(self.amortizer.mixing_logits, name)
        mixing_logits = mixing_logits(self.data).squeeze()
        mean_fields = {}
        for k in dist_params(prior):
            mean_field_amortizer = deep_getattr(self.amortizer.mean_fields,
                                                name + "." + k)
            mean_fields[k] = mean_field_amortizer(self.data)
        return mixing_logits, mean_fields

    def get_posterior(self, name: str, prior: Distribution) -> Distribution:
        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
            event_shape = prior.event_shape
            if isinstance(prior, dist.Independent):
                independent_ndims = prior.reinterpreted_batch_ndims
                prior = prior.base_dist
            else:
                independent_ndims = 0
            prior_params = dist_params(prior)
        alphas, lamdas = self._get_params(name, prior)
        alphas = torch.sigmoid(alphas)
        alphas = alphas.reshape(alphas.shape + (1,) * len(event_shape))
        for k, v in prior_params.items():
            param_transform = transform_to(prior.arg_constraints[k])
            lam = param_transform(lamdas[k])
            lam = lam.reshape(*lam.shape[:-1], *event_shape)
            prior_params[k] = alphas * v + (1 - alphas) * lam

        proposal = prior.__class__(**prior_params)
        if independent_ndims:
            proposal = dist.Independent(proposal, independent_ndims)
        posterior = dist.TransformedDistribution(proposal,
                                                 [transform.with_cache()])
        return posterior

    def _pyro_sample(self, msg):
        if msg["is_observed"] or site_is_subsample(msg):
            return
        prior = msg["fn"]
        msg["infer"]["prior"] = prior
        posterior = self.get_posterior(msg["name"], prior)
        if isinstance(posterior, torch.Tensor):
            posterior = dist.Delta(posterior, event_dim=prior.event_dim)
        if posterior.batch_shape != prior.batch_shape:
            posterior = posterior.expand(prior.batch_shape)
        msg["fn"] = posterior

    def _pyro_post_sample(self, msg):
        # Manually apply outer plates.
        prior = msg["infer"].get("prior")
        if prior is not None and prior.batch_shape != msg["fn"].batch_shape:
            msg["infer"]["prior"] = prior.expand(msg["fn"].batch_shape)
        return super()._pyro_post_sample(msg)

_msngrs = [
    AsviMessenger,
]

locals()["asvi"] = _make_handler(AsviMessenger, __name__)
