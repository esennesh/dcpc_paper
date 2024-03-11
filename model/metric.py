import torch

import utils

def ess(log_weight):
    log_ess = 2 * torch.logsumexp(log_weight, dim=0) -\
              torch.logsumexp(2 * log_weight, dim=0)
    return torch.exp(log_ess).mean().item()

def log_marginal(log_weight):
    return utils.logmeanexp(log_weight, 0, False).mean().item()

def log_joint(log_p):
    return log_p.mean().item()