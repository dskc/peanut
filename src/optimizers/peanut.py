# src/optimizers/peanut.py - Peanut v2, final version
# Adam with damped first step and inflated v init
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
            damp = group['first_step_damp']
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
                    state['v'] = grad.pow(2)
                    denom = state['v'].sqrt().add_(eps)
                    p.addcdiv_(grad, denom, value=-lr * damp)
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