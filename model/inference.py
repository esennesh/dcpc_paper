import functools
import math
import networkx as nx
from typing import Callable

import torch
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

def _resample(log_weights, estimate_normalizer=False):
    log_weights = torch.swapaxes(log_weights, 0, -1)
    discrete = dist.Categorical(logits=log_weights)
    indices = discrete.sample(sample_shape=torch.Size([log_weights.shape[-1]]))
    if estimate_normalizer:
        log_weights = torch.swapaxes(log_weights, 0, -1)
        log_normalizer = torch.logsumexp(log_weights, dim=0, keepdim=True) -\
                         math.log(log_weights.shape[0])
        return indices, log_normalizer
    return indices

def _ancestor_index(indices, tensor):
    resampled_tensor = []
    for b in range(indices.shape[1]):
        resampled_tensor.append(tensor[indices[:, b], b])
    return torch.stack(resampled_tensor, dim=1)

class PpcGraph:
    def __init__(self, temperature, trace=None):
        self._graph = nx.DiGraph()
        self._temperature = temperature
        self._trace = trace

    @property
    def temperature(self):
        return self._temperature

    @property
    def trace(self):
        return self._trace

    def set_kwargs(self, site, **kwargs):
        self._graph.nodes[site]['kwargs'] = kwargs

    def populate(self, trace):
        self._trace = trace
        self.trace.detach_()
        self.trace.compute_log_prob()
        self.log_prob.cache_clear()
        self.site_errors.cache_clear()
        self._log_complete_conditional.cache_clear()

    def add_node(self, site, parents, kernel):
        self._graph.add_node(site, kernel=kernel, kwargs={}, value=None)
        for parent in parents:
            self._graph.add_edge(parent, site)

    @functools.cache
    def log_prob(self, site, value, *args, **kwargs):
        assert len(args) == len(list(self._graph.predecessors(site)))
        args = [self.trace.nodes[parent]['value'] if arg is None else arg for
                (arg, parent) in zip(args, self._graph.predecessors(site))]
        proposal, entry = Trace(), self.trace.nodes[site].copy()
        if value is not None:
            entry['value'] = value
        proposal.add_node(site, **entry)
        kernel = functools.partial(self._graph.nodes[site]['kernel'],
                                   **self._graph.nodes[site]['kwargs'])
        with pyro.poutine.replay(trace=proposal):
            trace = pyro.poutine.trace(kernel).get_trace(*args, **kwargs)
        trace.compute_log_prob()
        return trace.nodes[site]['log_prob']

    @functools.cache
    def site_errors(self, site):
        value = self.trace.nodes[site]['value']
        parents = [self.trace.nodes[parent]['value'] for parent in
                   self._graph.predecessors(site)]
        if torch.is_floating_point(value):
            def logprobsum(value, *args, **kwargs):
                return self.log_prob(site, value, *args, **kwargs).sum()
            error = torch.func.grad(logprobsum,
                                    argnums=tuple(range(1+len(parents))))
            return error(value, *parents)
        raise NotImplementedError("Discrete prediction errors not implemented!")

    def complete_conditional_error(self, site):
        error = self.site_errors(site)[0]
        for child in self._graph.successors(site):
            site_index = list(self._graph.predecessors(child)).index(site)
            error = error + self.site_errors(child)[1 + site_index]
        return error

    @functools.cache
    def _log_complete_conditional(self, site, value):
        parents = (None,) * len(list(self._graph.predecessors(site)))
        log_site = self.log_prob(site, value, *parents)
        for child in self._graph.successors(site):
            parents = [value if s == site else None for s in
                       self._graph.predecessors(child)]
            log_site = log_site + self.log_prob(child, None, *parents)
        return log_site

    def update(self, site):
        if self.trace.nodes[site]['is_observed']:
            return (0., 0.)

        z = self.trace.nodes[site]['value']
        error = self.complete_conditional_error(site)
        proposal = dist.Normal(z + self.temperature * error, 2*self.temperature)
        proposal = proposal.to_event(self.trace.nodes[site]['fn'].event_dim)
        z_next = proposal.sample()

        log_cc = self._log_complete_conditional(site, z_next)
        log_proposal = proposal.log_prob(z_next)
        particle_indices, log_Zcc = _resample(log_cc - log_proposal,
                                              estimate_normalizer=True)
        z_next = _ancestor_index(particle_indices, z_next)

        self.trace.nodes[site]['value'] = z_next
        return (log_cc, log_Zcc)

    def update_sweep(self):
        log_proposal = 0.
        for site in self.trace.topological_sort():
            if site not in self._graph:
                continue
            log_cc, log_Zcc = self.update(site)
            log_proposal = log_proposal + log_Zcc - log_cc
        return self.trace, log_proposal

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
#    NeuralAsviMessenger,
]

for _msngr_cls in _msngrs:
    _handler_name, _handler = _make_handler(_msngr_cls)
    _handler.__module__ = __name__
    locals()[_handler_name] = _handler
