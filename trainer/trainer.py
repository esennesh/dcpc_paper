import functools
import math
import numpy as np
import pyro
from pyro.infer import SVI, JitTraceGraph_ELBO, TraceGraph_ELBO
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import utils

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 num_particles=4, jit=False):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.num_particles = num_particles
        self.jit = jit

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.jit:
            elbo = JitTraceGraph_ELBO(num_particles=self.num_particles,
                                      max_plate_nesting=1,
                                      vectorize_particles=True)
        else:
            elbo = TraceGraph_ELBO(num_particles=self.num_particles,
                                   max_plate_nesting=1,
                                   vectorize_particles=True)
        svi = SVI(self.model.model, self.model.guide, self.optimizer, loss=elbo)

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)
            loss = svi.step(data) / data.shape[0]

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(data))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss))
                if data.shape[1] == 1:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.jit:
            elbo = JitTraceGraph_ELBO(num_particles=self.num_particles,
                                      max_plate_nesting=1,
                                      vectorize_particles=True)
        else:
            elbo = TraceGraph_ELBO(num_particles=self.num_particles,
                                   max_plate_nesting=1,
                                   vectorize_particles=True)
        svi = SVI(self.model.model, self.model.guide, self.optimizer, loss=elbo)

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                loss = svi.evaluate_loss(data) / data.shape[0]

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(data))
                if data.shape[1] == 1:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

class PpcTrainer(BaseTrainer):
    """
    Particle Predictive Coding (PPC) Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 num_particles=4):
        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.num_particles = num_particles
        self._train_traces = [None] * len(self.data_loader)
        self._valid_traces = [None] * len(self.valid_data_loader)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _load_inference_state(self, batch_idx, data, train=True):
        trace = self._train_traces[batch_idx] if train\
                else self._valid_traces[batch_idx]
        if trace is None:
            with pyro.plate_stack('forward', (self.num_particles, len(data))):
                trace = pyro.poutine.trace(self.model).get_trace(data)
            trace.compute_log_prob()
        else:
            for site in trace:
                for key, val in trace.nodes[site].items():
                    if isinstance(val, torch.Tensor):
                        trace.nodes[site][key] = val.to(self.device)
            with pyro.plate_stack('forward', (self.num_particles, len(data))):
                trace = utils.regen_trace(self.model, trace, data)
        return trace

    def _save_inference_state(self, batch_idx, trace, train=True):
        for site, msg in list(trace.nodes.items()):
            if msg['type'] == 'sample' and not msg['is_observed']:
                msg['value'] = msg['value'].detach().cpu()

                saving_keys = {'type', 'name', 'infer', 'is_observed', 'value'}
                for k in set(msg.keys()) - saving_keys:
                    del msg[k]
            else:
                del trace.nodes[site]
        traces = self._train_traces if train else self._valid_traces
        traces[batch_idx] = trace

    def _ppc_step(self, i, data, train=True):
        trace = self._load_inference_state(i, data, train)

        # Wasserstein-gradient updates to latent variables
        self.model.graph.populate(trace)
        trace, log_weight = utils.importance(self.model.forward,
                                             self.model.guide, data)

        loss = (-log_weight).mean()
        if train:
            loss.backward()
            self.optimizer(pyro.get_param_store().values())
            pyro.infer.util.zero_grads(pyro.get_param_store().values())

        self._save_inference_state(i, trace, train)
        return loss, log_weight

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)

            loss, log_weight = self._ppc_step(batch_idx, data)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(log_weight))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                if len(data.shape) == 4:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        trace, log_weight = None, 0.

        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                loss, log_weight = self._ppc_step(batch_idx, data, False)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(log_weight))

                if len(data.shape) == 4:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
