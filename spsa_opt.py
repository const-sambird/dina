import torch
from torch.optim.optimizer import Optimizer
from qiskit_algorithms.optimizers import SPSA

class SPSAOptimiser(Optimizer):
    def __init__(self, params, maxiter=1, spsa_kwargs=None):
        defaults = {}
        super().__init__(params, defaults)
        self.spsa = SPSA(maxiter=maxiter, **(spsa_kwargs or {}))

    @torch.no_grad()
    def step(self, closure):
        loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                w = p.detach().cpu().numpy()

                # Define objective function for SPSA
                def obj(weights):
                    p.copy_(torch.tensor(weights, dtype=p.dtype, device=p.device))
                    return float(closure().item())

                result = self.spsa.minimize(fun=obj, x0=w)
                p.copy_(torch.tensor(result.x, dtype=p.dtype, device=p.device).view_as(p))

        return loss
