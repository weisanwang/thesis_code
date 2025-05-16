from copy import deepcopy
import torch
import torch.nn.functional as F

class CLStrategy:
    def on_domain_shift(self, model):
        """domain shift """
        pass
    def penalty(self, model, student_logits=None, inputs=None):
        """penalty loss"""
        return 0.0
    def after_backward(self, model):
        """updata"""
        pass

"""
"Elastic Weight Consolidation ++"

"""

def normalize_fn(fisher, EPS = 1e-20):
    return (fisher - fisher.min()) / (fisher.max() - fisher.min() + EPS)   

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

    def penalty(self, model, inputs=None):
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

"""
"Learning without Forgetting"

"""   

class LwFStrategy(CLStrategy):
    def __init__(self, device, lwf_lambda=1.0, temperature=2.0):
        self.device = device
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.model_old = None

    def on_domain_shift(self, model):
        # Save current model as old model
        self.model_old = deepcopy(model).eval().to(self.device)
        for p in self.model_old.parameters():
            p.requires_grad = False

    def penalty(self, model, student_logits=None, inputs=None):
        if self.model_old is None or inputs is None:
            return torch.tensor(0.0, device=self.device)

        # teacher logits
        with torch.no_grad():
            logits_old = self.model_old(inputs)

        # KL : KL(soft(teacher/T) || soft(student/T))
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(logits_old / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature * self.temperature)

        ###
        # log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        # q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        # res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")

        return self.lwf_lambda * loss_kl