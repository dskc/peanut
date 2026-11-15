# mlp_v23_check.py - Test fixed Peanut (Adam m) on MLP
# Same as v2.2 check, but uses src/optimizers/peanut.py fixed version
# MLP, Fashion-MNIST, batch 64, 10 epochs, 3 seeds (0,1,42), LR 0.01
# Output: results/mlp_v23_check.csv

import csv
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from optimizers.peanut import Peanut

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.net(x)

def run_one(seed, lr):
    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root='data', train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root='data', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    model = TinyMLP()
    opt = Peanut(model.parameters(), lr=lr)
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
    accs = []
    for seed in [0, 1, 42]:
        acc = run_one(seed, 0.01)
        accs.append(acc)
        print(f"Peanut v2.3 fixed seed {seed}: {acc:.4f}", flush=True)
    mean = sum(accs) / len(accs)
    worst = min(accs)
    print(f"Peanut v2.3 fixed MEAN {mean:.4f} WORST {worst:.4f}")
    with open('results/mlp_v23_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'mean', 'worst', 'seed0', 'seed1', 'seed42'])
        w.writerow(['Peanut v2.3 fixed', mean, worst] + accs)
    print("Saved to results/mlp_v23_check.csv")

if __name__ == '__main__':
    main()