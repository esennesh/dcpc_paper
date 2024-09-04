import functools
import lightning as L
import math
import numpy as np
import pyro
from pyro.infer import Importance, Predictive, SVI, JitTrace_ELBO, Trace_ELBO
import torch
import torch.nn.functional as F
import torchmetrics
from torchvision.utils import make_grid
from model import metric
from model.inference import ParticleDict, PpcGraphicalModel
from utils import inf_loop, MetricTracker
import utils

class LightningSvi(L.LightningModule):
    def __init__(self, importance, data: L.LightningDataModule, jit=False,
                 lr=1e-3, num_particles=4, cooldown=50, factor=0.9,
                 patience=100):
        super().__init__()
        self.cooldown = cooldown
        self.data = data
        self.factor = factor
        self.importance = importance
        self.lr = lr
        self.num_particles = num_particles
        self.patience = patience

        if jit:
            elbo = JitTrace_ELBO(num_particles=self.num_particles,
                                 max_plate_nesting=1,
                                 vectorize_particles=True)
        else:
            elbo = Trace_ELBO(num_particles=self.num_particles,
                              max_plate_nesting=1,
                              vectorize_particles=True)
        self.elbo = elbo(self.importance.model, self.importance.guide)
        self.predictive = Predictive(self.importance.model,
                                     guide=self.importance.guide,
                                     num_samples=self.num_particles,
                                     parallel=True)
        if len(self.data.dims) == 3 and self.data.dims[0] == 3:
            self.metrics = {
                "fid": torchmetrics.image.fid.FrechetInceptionDistance(
                    input_img_size=self.data.dims, normalize=True,
                )
            }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.importance.parameters(),
                                     amsgrad=True, lr=self.lr,
                                     weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, cooldown=self.cooldown, factor=self.factor,
            patience=self.patience
        )
        return {"lr_scheduler": lr_scheduler, "monitor": "valid/loss",
                "optimizer": optimizer}

    def forward(self, *args, mode="joint", **kwargs):
        if mode == "prior":
            with pyro.plate("importance", kwargs['P'], dim=-2):
                tp = pyro.poutine.trace(self.elbo.model).get_trace(*args,
                                                                   **kwargs)
        else:
            with pyro.poutine.uncondition():
                tp, tq = self.elbo.elbo._get_vectorized_trace(self.elbo.model,
                                                              self.elbo.guide,
                                                              args, kwargs)
        return tp.nodes['X']["fn"].base_dist.base_dist.loc

    @torch.no_grad()
    def test_step(self, batch, batch_idx, reset_fid=False):
        data, _, indices = batch
        data = data.to(self.device)
        loss = self.elbo(data)
        trace, tq = self.elbo.elbo._get_vectorized_trace(
            self.elbo.model, self.elbo.guide, (data,),
            {"B": len(data), "P": self.num_particles}
        )
        log_weight = utils.log_joint(trace) - utils.log_joint(tq)

        metrics = {
            "ess": metric.ess(trace, log_weight),
            "log_joint": metric.log_joint(trace, log_weight),
            "log_marginal": metric.log_marginal(trace, log_weight),
            "loss": loss
        }
        if len(self.data.dims) == 3 and self.data.dims[0] == 3:
            self.metrics['fid'] = self.metrics['fid'].to(self.device)
            self.metrics['fid'].update(self.data.reverse_transform(data),
                                       real=True)

            B = len(data)
            imgs = self.forward(data, B=B, mode="prior", P=self.num_particles)
            imgs = self.data.reverse_transform(
                imgs.view(self.num_particles*B, *self.data.dims)
            ).clamp(0, 1)
            self.metrics['fid'].update(imgs, real=False)

            if reset_fid:
                metrics["fid"] = self.metrics['fid'].compute()
                self.metrics['fid'].reset()
            self.metrics['fid'] = self.metrics['fid'].cpu()
            del imgs
        del data

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training logic for an epoch

        :param batch: Batch of training data for current training epoch.
        :return: Loss in this epoch.
        """
        data, target, _ = batch
        loss = self.elbo(data)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation logic for an epoch

        :param batch: Batch of training data for current validation epoch.
        :return: Loss in this epoch.
        """
        data, target, _ = batch
        loss = self.elbo(data)
        self.log("valid/loss", loss, sync_dist=True)
        return loss

class LightningPpc(L.LightningModule):
    """
    Lightning module for Population Predictive Coding (PPC)
    """
    def __init__(self, graph: PpcGraphicalModel, data: L.LightningDataModule,
                 cooldown=50, factor=0.9, lr=1e-3, num_particles=4,
                 num_sweeps=1, patience=100):
        super().__init__()
        self.save_hyperparameters(ignore=["data", "graph"])
        self.cooldown = cooldown
        self.data = data
        self.factor = factor
        self.lr = lr
        self.graph = graph
        self.metrics = {}
        self.num_particles = num_particles
        self.num_sweeps = num_sweeps
        self.patience = patience
        self.predictive = Predictive(self.graph.model, guide=self.graph.guide,
                                     num_samples=self.num_particles)

        self._num_train = len(data.train_dataloader().dataset)
        self._num_valid = len(data.val_dataloader().dataset)
        if len(self.data.dims) == 3 and self.data.dims[0] == 3:
            self.metrics['fid'] =\
                torchmetrics.image.fid.FrechetInceptionDistance(
                    input_img_size=self.data.dims, normalize=True,
                )

    def setup(self, stage):
        num_train, num_valid = self._num_train, self._num_valid
        self.particles = {
            "train": ParticleDict(num_train, self.num_particles),
            "valid": ParticleDict(num_valid, self.num_particles)
        }
        self.graph.to(self.device)
        for batch_idx, batch in enumerate(self.data.train_dataloader()):
            batch = (batch[0].to(self.device), batch[1],
                     batch[2].to(self.device))
            self._initialize_particles(batch, batch_idx)
        for batch_idx, batch in enumerate(self.data.val_dataloader()):
            batch = (batch[0].to(self.device), batch[1],
                     batch[2].to(self.device))
            self._initialize_particles(batch, batch_idx, False)

    def _initialize_particles(self, batch, batch_idx, train=True):
        data, target, indices = batch
        with self.graph.condition(**self.graph.conditioner(data)) as graph:
            graph(B=data.shape[0], lr=self.lr/self.num_particles, mode="prior",
                  P=self.num_particles)
            self._save_particles(indices, train)

    def _load_particles(self, indices, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in particles:
            particle_vals = particles.get_particles(site, indices)
            if self.device != particle_vals.device:
                particle_vals = particle_vals.to(self.device)
            self.graph.update(site, particle_vals)

    def _save_particles(self, indices, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in self.graph.stochastic_nodes:
            particle_vals = self.graph.nodes[site]['value'].detach()
            particles.set_particles(site, indices,
                                    particle_vals.to(device='cpu'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph.parameters(), amsgrad=True,
                                     lr=self.lr, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, cooldown=self.cooldown, factor=self.factor,
            patience=self.patience
        )
        return {"lr_scheduler": lr_scheduler, "monitor": "valid/loss",
                "optimizer": optimizer}

    def forward(self, *args, **kwargs):
        return self.predictive(*args, **kwargs)

    def on_load_checkpoint(self, checkpoint):
        self.particles = checkpoint["particle_dicts"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["particle_dicts"] = self.particles

    def ppc_step(self, data):
        with self.graph.condition(**self.graph.conditioner(data)) as graph:
            for _ in range(self.num_sweeps - 1):
                graph(B=data.shape[0], lr=self.lr, P=self.num_particles)
            return graph(B=data.shape[0], lr=self.lr, P=self.num_particles)

    @torch.no_grad()
    def test_step(self, batch, batch_idx, reset_fid=False):
        data, _, indices = batch
        data = data.to(self.device)
        self.graph.clear()
        with self.graph.condition(**self.graph.conditioner(data)) as graph:
            graph(B=data.shape[0], mode="prior", P=self.num_particles)
        trace, log_weight = self.ppc_step(data)
        self.graph.clear()

        metrics = {
            "ess": metric.ess(trace, log_weight),
            "log_joint": metric.log_joint(trace, log_weight),
            "log_marginal": metric.log_marginal(trace, log_weight),
            "loss": -log_weight.mean()
        }
        if len(self.data.dims) == 3 and self.data.dims[0] == 3:
            self.metrics['fid'] = self.metrics['fid'].to(self.device)
            self.metrics['fid'].update(self.data.reverse_transform(data),
                                       real=True)

            posterior = {k: torch.cat((v, self.particles["valid"][k]), dim=1)
                            for k, v in self.particles["train"].items()}
            B = len(data) // self.num_particles
            imgs = self.graph.predict(B=B, P=self.num_particles, **posterior)
            imgs = self.data.reverse_transform(
                imgs.view(self.num_particles*B, *self.data.dims)
            ).clamp(0, 1)
            self.metrics['fid'].update(imgs, real=False)

            if reset_fid:
                metrics["fid"] = self.metrics['fid'].compute()
                self.metrics['fid'].reset()
            self.metrics['fid'] = self.metrics['fid'].cpu()
            del imgs
        del data

        return metrics

    def training_step(self, batch, batch_idx):
        data, _, indices = batch
        self._load_particles(indices, train=True)
        trace, log_weight = self.ppc_step(data)
        loss = (F.softmax(log_weight, dim=0).detach() * log_weight).sum(dim=0)
        loss = -loss.mean()
        self._save_particles(indices, train=True)

        self.log("train/ess", metric.ess(trace, log_weight.detach()))
        self.log("train/log_joint", metric.log_joint(trace,
                                                     log_weight.detach()))
        self.log("train/log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()))
        self.log("train/loss", loss)
        return loss * self._num_train

    def validation_step(self, batch, batch_idx):
        data, _, indices = batch
        self._load_particles(indices, train=False)
        trace, log_weight = self.ppc_step(data)
        loss = (F.softmax(log_weight, dim=0).detach() * log_weight).sum(dim=0)
        loss = -loss.mean()
        self._save_particles(indices, train=False)

        self.log("valid/ess", metric.ess(trace, log_weight.detach()),
                 sync_dist=True)
        self.log("valid/log_joint", metric.log_joint(trace,
                                                     log_weight.detach()),
                 sync_dist=True)
        self.log("valid/log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()),
                 sync_dist=True)
        self.log("valid/loss", loss, sync_dist=True)
        return loss
