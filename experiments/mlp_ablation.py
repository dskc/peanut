# experiments/mlp_ablation.py - Ablation + Warmup
# 5 methods, 2 LRs, 3 seeds, MLP, 10 epochs
# Methods: adam, adam_warmup, only_damp, only_inflated_v, both_peanut_v2
import torch
import torch.nn as nn
import sys
import os
import csv
import time
sys.path.append("src")
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers.peanut import Peanut
import numpy as np

torch.set_num_threads(10)
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28*28,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,10))
    def forward(self, x):
        return self.net(x)

# Only damp: normal v init, damped first step
from torch.optim import Optimizer
class OnlyDamp(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    @ torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']; beta1, beta2 = group['betas']; eps = group['eps']; wd = group['weight_decay']
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
                    state['v'] = grad.pow(2).mul_(1 - beta2)
                    denom = state['v'].sqrt().add_(eps)
                    p.addcdiv_(grad, denom, value=-lr * 0.2)
                    continue
                state['step'] += 1
                state['m'].mul_(beta1).add_(grad, alpha=1-beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                bias1 = 1 - beta1 ** state['step']; bias2 = 1 - beta2 ** state['step']
                m_hat = state['m'] / bias1; v_hat = state['v'] / bias2
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss

# Only inflated v: inflated v init, no damped first step
class OnlyInflatedV(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    @ torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']; beta1, beta2 = group['betas']; eps = group['eps']; wd = group['weight_decay']
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
                    p.addcdiv_(grad, denom, value=-lr * 1.0)
                    continue
                state['step'] += 1
                state['m'].mul_(beta1).add_(grad, alpha=1-beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                bias1 = 1 - beta1 ** state['step']; bias2 = 1 - beta2 ** state['step']
                m_hat = state['m'] / bias1; v_hat = state['v'] / bias2
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss

def train_one(name, seed=0, epochs=10, lr=0.01):
    torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=g)
    model = TinyMLP()
    loss_fn = nn.CrossEntropyLoss()
    warmup_steps = 100
    step_count = 0
    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "adam_warmup":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif name == "only_damp":
        opt = OnlyDamp(model.parameters(), lr=lr)
    elif name == "only_inflated_v":
        opt = OnlyInflatedV(model.parameters(), lr=lr)
    else:
        opt = Peanut(model.parameters(), lr=lr, first_step_damp=0.2)
    accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            # linear warmup for adam_warmup only
            if name == "adam_warmup":
                step_count += 1
                if step_count <= warmup_steps:
                    warm_lr = lr * step_count / warmup_steps
                    for pg in opt.param_groups:
                        pg['lr'] = warm_lr
                else:
                    for pg in opt.param_groups:
                        pg['lr'] = lr
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                correct += (model(x).argmax(1)==y).sum().item()
        acc = correct/len(test_data)
        accs.append(acc)
    return accs

if __name__ == "__main__":
    seeds = [0, 1, 42]
    lrs = [0.003, 0.01]
    methods = ["adam", "adam_warmup", "only_damp", "only_inflated_v", "both_peanut_v2"]
    os.makedirs("results", exist_ok=True)
    with open("results/mlp_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "lr", "seed0", "seed1", "seed42", "mean", "std", "worst_seed"])
        for lr in lrs:
            for n in methods:
                finals = []
                for s in seeds:
                    accs = train_one(n, seed=s, epochs=10, lr=lr)
                    finals.append(accs[-1])
                    print(f"{n} lr={lr} seed={s} final={accs[-1]:.4f}", flush=True)
                w.writerow([n, lr, finals[0], finals[1], finals[2], np.mean(finals), np.std(finals), min(finals)])
                print(f"{n} lr={lr} mean={np.mean(finals):.4f} std={np.std(finals):.4f} worst={min(finals):.4f}", flush=True)
    print("Saved: results/mlp_ablation.csv", flush=True)