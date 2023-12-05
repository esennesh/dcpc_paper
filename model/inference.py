from typing import Callable

import torch
from torch.distributions import biject_to, transform_to
import torch.nn as nn

import pyro.distributions as dist
from pyro.distributions.distribution import Distribution
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr, helpful_support_errors, _product
from pyro.nn.module import PyroModule, PyroParam
from pyro.poutine.handlers import _make_handler
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.util import site_is_subsample

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
