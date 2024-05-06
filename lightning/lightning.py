import functools
import math
import numpy as np
import pyro
from pyro.optim import lr_scheduler
from pyro.infer import SVI, JitTraceGraph_ELBO, TraceGraph_ELBO
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.inference import ParticleDict
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
    def __init__(self, model, metric_ftns, optimizer, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None,
                 len_epoch=None, num_particles=4, num_sweeps=1):
        resume = config.resume
        if config.resume is not None:
            config.resume = None
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
        self.num_sweeps = num_sweeps
        self.train_particles = ParticleDict(len(self.data_loader.sampler),
                                            num_particles)
        self.valid_particles = ParticleDict(len(self.valid_data_loader.sampler),
                                            num_particles)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        for batch_idx, (data, target, batch_indices) in enumerate(self.data_loader):
            data = data.to(self.device)
            self._initialize_particles(batch_indices, data)
            self.logger.debug("Initialize particles: train batch {}".format(batch_idx))
        self.model.clear()

        for batch_idx, (data, target, batch_indices) in enumerate(self.valid_data_loader):
            data = data.to(self.device)
            self._initialize_particles(batch_indices, data, False)
            self.logger.debug("Initialize particles: valid batch {}".format(batch_idx))
        self.model.clear()

        if resume is not None:
            self._resume_checkpoint(resume)

    @property
    def module(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def train(self, profiler=None):
        self.train_particles = self.train_particles.to(self.device)
        self.valid_particles = self.valid_particles.to(self.device)
        super().train(profiler=profiler)

    def _initialize_particles(self, batch_indices, data, train=True):
        data_loader = self.data_loader if train else self.valid_data_loader
        self.model(data, prior=True, P=self.num_particles)
        self._save_particles(batch_indices, train)

    def _load_particles(self, batch_indices, train=True):
        particles = self.train_particles if train else self.valid_particles
        for site in particles:
            value = particles.get_particles(site, batch_indices)
            self.module.update(site, value.to(self.device))

    def _save_particles(self, batch_indices, train=True):
        particles = self.train_particles if train else self.valid_particles
        for site in self.module.stochastic_nodes:
            value = self.module.nodes[site]['value'].detach()
            particles.set_particles(site, batch_indices, value)

    def _ppc_step(self, batch_indices, data, train=True):
        data_loader = self.data_loader if train else self.valid_data_loader
        self._load_particles(batch_indices, train)

        # Wasserstein-gradient updates to latent variables
        for _ in range(self.num_sweeps - 1):
            self.model(data, lr=self.lr, P=self.num_particles)
        trace, log_weight = self.model(data, lr=self.lr, P=self.num_particles)
        trace.detach_()

        loss = -log_weight.mean()
        if train:
            (loss * len(data_loader.dataset)).backward()
            self.optimizer(pyro.get_param_store().values())
            pyro.infer.util.zero_grads(pyro.get_param_store().values())

        self._save_particles(batch_indices, train)
        return loss, trace, log_weight.detach()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_particles.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, batch_indices) in enumerate(self.data_loader):
            data = data.to(self.device)
            loss, trace, log_weight = self._ppc_step(batch_indices, data)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(trace, log_weight))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                if data.shape[1] == 1:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            if isinstance(self.optimizer, lr_scheduler.PyroLRScheduler):
                self.optimizer.step(val_log['loss'])

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
        self.valid_particles.train()
        with torch.no_grad():
            for batch_idx, (data, target, batch_indices) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                loss, trace, log_weight = self._ppc_step(batch_indices, data, False)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(trace, log_weight))

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

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'train_particles': self.train_particles.state_dict(),
            'valid_particles': self.valid_particles.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path,map_location=torch.device('cpu'))
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.train_particles.load_state_dict(checkpoint['train_particles'])
        self.valid_particles.load_state_dict(checkpoint['valid_particles'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
