from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import pyro
import torch
import utils

class BaseModel(pyro.nn.PyroModule):
    """
    Base class for all models
    """
    def forward(self, *args, **kwargs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])

class ImportanceModel(BaseModel):
    def forward(self, *args, B=1, prior=False, P=1, **kwargs):
        with pyro.plate_stack("importance", (P, B)):
            if prior:
                return self.model(*args, **kwargs)
            return utils.importance(self.generate, self.guide, *args, **kwargs)

    @abstractmethod
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def guide(self, *args, **kwargs):
        raise NotImplementedError

class MarkovKernel(pyro.nn.PyroModule):
    """
    Base class for Markov kernels that output a Pyro distribution
    """
    @property
    def event_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> pyro.distributions.Distribution:
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])

@dataclass
class MarkovKernelApplication:
    kernel: str
    args: tuple
    kwargs: dict
