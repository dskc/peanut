# mlp_m_test.py - Does zero m matter?
# Arm 1: Peanut as shipped - m=0 on step 1, damp 0.2, v=g1^2
# Arm 2: Peanut with Adam m - m=Adam m on step 1, damp 0.2, v=g1^2
# MLP, Fashion-MNIST, batch 64, 10 epochs, 3 seeds (0,1,42), LR 0.01
# Output: results/mlp_m_test.csv

import csv
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Model: same TinyMLP as before ---
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

# --- Peanut variants, only m differs on step 1 ---
class PeanutMTest(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, use_adam_m=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.use_adam_m = use_adam_m  # False = shipped, True = Adam m

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                m, v = state['m'], state['v']

                if step == 1:
                    # v: Peanut inflated v in both arms
                    v.copy_(g * g)
                    if self.use_adam_m:
                        # Arm 2: Adam m
                        m.copy_((1 - beta1) * g)
                        m_hat = m / (1 - beta1 ** step)
                        update = m_hat / (v.sqrt() + eps)
                    else:
                        # Arm 1: Shipped, m = 0, update from g directly with damp
                        # This matches shipped Peanut: first update = damp * sign(g)
                        update = g / (v.sqrt() + eps)
                    # Damped first step, both arms
                    p.add_(update, alpha=-lr * 0.2)
                else:
                    # Normal Adam from step 2 onward, both arms
                    # For Arm 1, m starts from 0 here, so step 2 m = (1-beta1)*g2
                    # For Arm 2, m continues from step 1 m
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    m_hat = m / (1 - beta1 ** step)
                    v_hat = v / (1 - beta2 ** step)
                    update = m_hat / (v_hat.sqrt() + eps)
                    p.add_(update, alpha=-lr)
        return None

def run_one(seed, use_adam_m):
    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root='data', train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root='data', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = TinyMLP()
    opt = PeanutMTest(model.parameters(), lr=0.01, use_adam_m=use_adam_m)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def main():
    os.makedirs('results', exist_ok=True)
    rows = []
    for use_adam_m in [False, True]:
        name = 'Peanut_Adam_m' if use_adam_m else 'Peanut_shipped_zero_m'
        accs = []
        for seed in [0, 1, 42]:
            acc = run_one(seed, use_adam_m)
            accs.append(acc)
            print(f"{name} seed {seed}: {acc:.4f}", flush=True)
        mean = sum(accs) / len(accs)
        worst = min(accs)
        print(f"{name} MEAN {mean:.4f} WORST {worst:.4f}")
        rows.append([name, mean, worst] + accs)

    with open('results/mlp_m_test.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'mean', 'worst', 'seed0', 'seed1', 'seed42'])
        w.writerows(rows)
    print("Saved to results/mlp_m_test.csv")

if __name__ == '__main__':
    main()