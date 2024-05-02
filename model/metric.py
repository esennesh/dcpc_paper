import torch

import utils

def ess(trace, log_weight):
    log_ess = 2 * torch.logsumexp(log_weight, dim=0) -\
              torch.logsumexp(2 * log_weight, dim=0)
    return torch.exp(log_ess).mean().item()

def log_marginal(trace, log_weight):
    return utils.logmeanexp(log_weight, 0, False).mean().item()

def log_joint(trace, log_weight):
    return utils.log_joint(trace).mean().item()
