"""
EWC++ from paper "Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence"

"""
from copy import deepcopy
import torch

def normalize_fn(fisher, EPS = 1e-20):
    return (fisher - fisher.min()) / (fisher.max() - fisher.min() + EPS)

class EWCpp(object):
    def __init__(self, model, model_old, device, alpha=0.9, fisher=None, normalize=True):

        self.model = model
        self.model_old = model_old
        self.model_old_dict = self.model_old.state_dict()

        self.device = device
        self.alpha = alpha
        self.normalize = normalize
        
        if fisher is not None: # initialize as old Fisher Matrix
            self.fisher_old = fisher
            for key in self.fisher_old:
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = self.fisher_old[key].to(device)
            self.fisher = deepcopy(fisher)
            if normalize:
                self.fisher_old = {n: normalize_fn(self.fisher_old[n]) for n in self.fisher_old}

        else: # initialize a new Fisher Matrix
            self.fisher_old = None
            self.fisher = {n:torch.zeros_like(p, device=device, requires_grad=False) 
                           for n, p in self.model.named_parameters() if p.requires_grad} 

    def update(self):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = (self.alpha * p.grad.data.pow(2)) + ((1-self.alpha)*self.fisher[n])

    def get_fisher(self):
        return self.fisher # return the new Fisher matrix

    def penalty(self):
        loss = 0
        if self.fisher_old is None:
            return 0.
        for n, p in self.model.named_parameters():
            loss += (self.fisher_old[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss