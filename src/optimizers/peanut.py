# src/optimizers/peanut.py - Peanut v2.2, both dampings LR-aware
# Best version: mean wins at both 0.003 and 0.01
import torch
from torch.optim import Optimizer

class Peanut(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, first_step_damp=0.2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, first_step_damp=first_step_damp))

    @ torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            base_damp = group['first_step_damp']
            effective_damp = base_damp * 0.01 / lr
            if effective_damp > 1.0:
                effective_damp = 1.0
            if effective_damp < 0.05:
                effective_damp = 0.05
            inflation = lr / 0.01
            if inflation > 1.0:
                inflation = 1.0
            if inflation < 0.0:
                inflation = 0.0
            v_factor = (1 - beta2) + beta2 * inflation
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 1
                    state['m'] = torch.zeros_like(p)
                    state['v'] = grad.pow(2).mul_(v_factor)
                    denom = state['v'].sqrt().add_(eps)
                    p.addcdiv_(grad, denom, value=-lr * effective_damp)
                    continue

                state['step'] += 1
                state['m'].mul_(beta1).add_(grad, alpha=1-beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']
                m_hat = state['m'] / bias1
                v_hat = state['v'] / bias2
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss