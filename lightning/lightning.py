import functools
import lightning as L
import math
import numpy as np
import pyro
from pyro.infer import Importance, Predictive, SVI, JitTraceGraph_ELBO, TraceGraph_ELBO
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model import metric
from model.inference import ParticleDict
from utils import inf_loop, MetricTracker
import utils

class LightningSvi(L.LightningModule):
    def __init__(self, importance, jit=False, lr=1e-3, num_particles=4):
        super().__init__()
        self.importance = importance
        self.lr = lr
        self.num_particles = num_particles

        if jit:
            elbo = JitTraceGraph_ELBO(num_particles=self.num_particles,
                                      max_plate_nesting=1,
                                      vectorize_particles=True)
        else:
            elbo = TraceGraph_ELBO(num_particles=self.num_particles,
                                   max_plate_nesting=1,
                                   vectorize_particles=True)
        self.elbo = elbo(self.importance.model, self.importance.guide)
        self.predictive = Predictive(self.importance.model,
                                     guide=self.importance.guide,
                                     num_samples=self.num_particles)

    def configure_optimizers(self):
        return torch.optim.Adam(self.elbo.parameters(), amsgrad=True,
                                lr=self.lr, weight_decay=0.)

    def forward(self, *args, **kwargs):
        return self.predictive(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        Training logic for an epoch

        :param batch: Batch of training data for current training epoch.
        :return: Loss in this epoch.
        """
        data, target = batch
        loss = self.elbo(data)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation logic for an epoch

        :param batch: Batch of training data for current validation epoch.
        :return: Loss in this epoch.
        """
        data, target = batch
        loss = self.elbo(data)
        self.log("train_loss", loss)
        return loss

class LightningPpc(L.LightningModule):
    """
    Lightning module for Population Predictive Coding (PPC)
    """
    def __init__(self, model: PpcGraphicalModel, num_train, num_valid,
                 cooldown=50, factor=0.9, lr=1e-3, num_particles=4,
                 num_sweeps=1, patience=100):
        super().__init__()
        self.cooldown = cooldown
        self.factor = factor
        self.lr = lr
        self.model = model
        self.num_particles = num_particles
        self.num_sweeps = num_sweeps
        self.patience = patience
        self.predictive = Predictive(self.model.model, guide=self.model.guide,
                                     num_samples=self.num_particles)

        self.particles = {"train": ParticleDict(num_train, num_particles),
                          "valid": ParticleDict(num_valid, num_particles)}

    def _initialize_particles(self, batch, batch_idx, train=True):
        data, target = batch
        self.model.model(data, P=self.num_particles)
        self._save_particles(batch_idx, train)

    def _load_particles(self, batch, batch_idx, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in particles:
            self.model.update(site, particles.get_particles(site, batch_idx))

    def _save_particles(self, batch, batch_idx, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in self.model.stochastic_nodes:
            particles.set_particles(site, batch_idx,
                                    self.model.nodes[site]['value'].detach())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.elbo.parameters(), amsgrad=True,
                                     lr=self.lr, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, cooldown=self.cooldown, factor=self.factor,
            patience=self.patience
        )
        return {"lr_scheduler": lr_scheduler, "optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        self.particles = checkpoint["particle_dicts"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["particle_dicts"] = self.particles

    def ppc_step(self, batch, batch_idx):
        data, target = batch
        for _ in range(self.num_sweeps - 1):
            pyro.infer.importance.vectorized_importance_weights(
                self.model.model, self.model.guide, data, lr=self.lr,
                num_samples=self.num_particles
            )
        log_weight, ptrace, qtrace =\\
            pyro.infer.importance.vectorized_importance_weights(
                self.model.model, self.model.guide, data, lr=self.lr,
                num_samples=self.num_particles
            )
        return log_weight, ptrace

    def training_step(self, batch, batch_idx):
        self._load_particles(batch_idx, train=True)
        log_weight, trace = self.ppc_step(batch, batch_idx)
        loss = -log_weight.mean()
        self._save_particles(batch_idx, train=True)

        self.log("train_ess", metric.ess(trace, log_weight.detach()))
        self.log("train_log_joint", metric.log_joint(trace,
                                                     log_weight.detach()))
        self.log("train_log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._load_particles(batch_idx, train=False)
        log_weight, trace = self.ppc_step(batch, batch_idx)
        loss = -log_weight.mean()
        self._save_particles(batch_idx, train=False)

        self.log("train_ess", metric.ess(trace, log_weight.detach()))
        self.log("train_log_joint", metric.log_joint(trace,
                                                     log_weight.detach()))
        self.log("train_log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()))
        self.log("train_loss", loss)
        return loss
