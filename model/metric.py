import torch

import utils

def ess(log_weight):
    log_normalized = log_weight - torch.logsumexp(log_weight, dim=0)
    return torch.exp(-torch.logsumexp(2 * log_normalized, dim=0)).mean().item()

def log_marginal(log_weight):
    return utils.logmeanexp(log_weight, 0, False).mean().item()
