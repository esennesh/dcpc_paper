import functools
import math
import networkx as nx
from typing import Callable, Sequence

import torch
import torch.distributions.constraints as constraints
from torch.distributions import biject_to, transform_to
import torch.nn as nn

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

def _resample(log_weights, estimate_normalizer=False):
    discrete = dist.Categorical(logits=torch.swapaxes(log_weights, 0, -1))
    indices = discrete.sample(sample_shape=torch.Size([log_weights.shape[0]]))
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

    def get_particles(self, key: str, idx: Sequence[int]) -> torch.Tensor:
        val = self[key]
        assert val.shape[self._batch_dim] == self.num_data
        assert val.shape[self._particle_dim] == self.num_particles
        return torch.index_select(val, self._batch_dim, torch.LongTensor(idx))

    def set_particles(self, key: str, idx: Sequence[int], val: torch.Tensor):
        assert val.shape[self._particle_dim] == self.num_particles
        if key not in self:
            shape = list(val.shape)
            shape[self._batch_dim] = self.num_data
            self[key] = torch.zeros(*shape)
        with torch.no_grad():
            indices = torch.LongTensor(idx).view(
                (1,) * self._batch_dim + (len(idx),) +\
                (1,) * len(val.shape[self._batch_dim+1:])
            )
            self[key].scatter_(self._batch_dim, indices.expand(val.shape),
                               val.to(self[key].device))

class PpcMessenger(TraceMessenger):
    def __init__(self, graph: GraphicalModel, temperature):
        self._graph = graph
        self._temperature = temperature
        super().__init__()

    def _compute_site_errors(self, site):
        value = self.graph.nodes[site]['value']
        pvals = self.graph.parent_vals(site)
        def logprobsum(value, *args, **kwargs):
            return self.graph.log_prob(site, value, *args, **kwargs).sum()

        if torch.is_floating_point(value):
            error = torch.func.grad(logprobsum,
                                    argnums=tuple(range(1+len(pvals))))
        else:
            raise NotImplementedError("Discrete prediction errors not implemented!")
        return error(value, *pvals)

    def _complete_conditional_error(self, site):
        error = self._site_errors(site)[0]
        for child in self.graph.child_sites(site):
            site_index = list(self.graph.parent_sites(child)).index(site)
            error = error + self._site_errors(child)[1 + site_index]
        return error

    def _log_complete_conditional(self, site, value):
        args = tuple(self.graph.nodes[p]['value'] for p in
                     self.graph.parent_sites(site))
        log_sitecc = self.graph.log_prob(site, value, *args)
        for child in self.graph.child_sites(site):
            args = tuple(value if s == site else self.graph.nodes[s]['value']
                         for s in self.graph.parent_sites(child))
            log_sitecc = log_sitecc + self.graph.log_prob(
                child, self.graph.nodes[child]['value'], *args
            )
        return log_sitecc

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

    def _site_errors(self, site):
        if self.graph.nodes[site].get('errors', None) is None:
            self.graph.nodes[site]['errors'] = self._compute_site_errors(site)
        return self.graph.nodes[site]['errors']

    def get_posterior(self, name: str, prior: Distribution) -> Distribution:
        with torch.no_grad():
            z = self.graph.nodes[name]['value']
            error = self._complete_conditional_error(name)
            proposal = dist.Normal(z + self.temperature * error,
                                   math.sqrt(2*self.temperature))
            proposal = proposal.to_event(self.graph.nodes[name]['event_dim'])
            z_next = proposal.sample()

            log_cc = self._log_complete_conditional(name, z_next)
            log_proposal = proposal.log_prob(z_next)
            particle_indices, log_Zcc = _resample(log_cc - log_proposal,
                                                  estimate_normalizer=True)
            z_next = _ancestor_index(particle_indices, z_next)
            log_cc = _ancestor_index(particle_indices, log_cc)

            self.graph.nodes[name]['errors'] = None
            for child in self.graph.child_sites(name):
                self.graph.nodes[child]['errors'] = None

            return dist.Delta(z_next, log_cc - log_Zcc,
                              event_dim=self.graph.nodes[name]['event_dim'])

    def get_traces(self):
        guide_trace = pyro.poutine.util.prune_subsample_sites(self.trace)
        model_trace = model_trace = guide_trace.copy()
        for name, guide_site in list(guide_trace.nodes.items()):
            if guide_site["type"] != "sample" or guide_site["is_observed"]:
                del guide_trace.nodes[name]
                continue
            model_site = model_trace.nodes[name].copy()
            model_site["fn"] = guide_site["infer"]["prior"]
            model_trace.nodes[name] = model_site
        return model_trace, guide_trace

    @property
    def graph(self):
        return self._graph

    @property
    def temperature(self):
        return self._temperature

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
    PpcMessenger,
#    NeuralAsviMessenger,
]

for _msngr_cls in _msngrs:
    _handler_name, _handler = _make_handler(_msngr_cls)
    _handler.__module__ = __name__
    locals()[_handler_name] = _handler
