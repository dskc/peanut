# experiments/mlp_v21_check.py - Test LR-aware Peanut, MLP, 10 epochs, 2 LRs
# Includes start time and stop time
import torch
import torch.nn as nn
import sys
import os
import csv
import time
import datetime
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

def train_one(name, seed=0, epochs=10, lr=0.01):
    torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=g)
    model = TinyMLP()
    loss_fn = nn.CrossEntropyLoss()
    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = Peanut(model.parameters(), lr=lr, first_step_damp=0.2)
    accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
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
    run_start = time.perf_counter()
    start_wall = datetime.datetime.now()
    print(f"START TIME: {start_wall.strftime('%H:%M:%S')}", flush=True)
    seeds = [0, 1, 42]
    lrs = [0.003, 0.01]
    os.makedirs("results", exist_ok=True)
    with open("results/mlp_v21_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "lr", "seed0", "seed1", "seed42", "mean", "std", "worst_seed"])
        for lr in lrs:
            for n in ["adam", "peanut_v21"]:
                finals = []
                for s in seeds:
                    accs = train_one(n, seed=s, epochs=10, lr=lr)
                    finals.append(accs[-1])
                    print(f"{n} lr={lr} seed={s} final={accs[-1]:.4f}", flush=True)
                w.writerow([n, lr, finals[0], finals[1], finals[2], np.mean(finals), np.std(finals), min(finals)])
                print(f"{n} lr={lr} mean={np.mean(finals):.4f} std={np.std(finals):.4f} worst={min(finals):.4f}", flush=True)
    run_end = time.perf_counter()
    end_wall = datetime.datetime.now()
    total_sec = run_end - run_start
    print(f"STOP TIME: {end_wall.strftime('%H:%M:%S')}", flush=True)
    print(f"START: {start_wall.strftime('%H:%M:%S')} STOP: {end_wall.strftime('%H:%M:%S')} TOTAL: {total_sec:.1f}s", flush=True)
    print("Saved: results/mlp_v21_check.csv", flush=True)