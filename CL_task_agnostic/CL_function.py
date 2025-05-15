"""
EWC++ from paper "Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence"

"""
from copy import deepcopy
import torch

def normalize_fn(fisher, EPS = 1e-20):
    return (fisher - fisher.min()) / (fisher.max() - fisher.min() + EPS)   

class CLStrategy:
    def on_domain_shift(self, model):
        """domain shift """
        pass
    def penalty(self, model):
        """penalty loss"""
        return 0.0
    def after_backward(self, model):
        """updata"""
        pass


class EWCPlusStrategy(CLStrategy):
    def __init__(self, device, alpha=0.9, ewc_lambda=1.0, normalize=True):
        self.device = device
        self.alpha = alpha
        self.ewc_lambda = ewc_lambda
        self.normalize = normalize
        self.model_old = None
        self.model_old_dict = None
        self.fisher_old = None
        self.fisher_curr = {
            n: torch.zeros_like(p, device=self.device, requires_grad=False)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def on_domain_shift(self, model):
        # If had a fisher matrix already, then save it as fisher_old to calculate penalty
        if self.fisher_curr is not None:
            self.fisher_old = {
                n: (normalize_fn(f) if self.normalize else f).detach().clone()
                for n, f in self.fisher_curr.items()
            }
            self.model_old_dict = {
                n: p.detach().clone().to(self.device)
                for n, p in model.named_parameters()
                if p.requires_grad
                }

    def penalty(self, model):
        if self.fisher_old is None:
            return 0.0
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                fi = self.fisher_old[n]
                po = self.model_old_dict[n]
                loss += (fi * (p - po).pow(2)).sum()
        return self.ewc_lambda * loss

    def after_backward(self, model):
        # Accumulate Fisher matrix after each backward pass
        for n, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                self.fisher_curr[n] = (
                    self.alpha * p.grad.detach().pow(2) + (1-self.alpha) * self.fisher_curr[n]
                    )