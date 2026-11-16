# mnist_baseline.py - Milestone 1, MNIST 9-run baseline
# MNIST, Small MLP, Adam vs AdamW vs Peanut, 3 seeds, 15 epochs, batch 64, CPU only
# Metrics: Accuracy, loss, time, RAM
# Total: 9 runs. Run 1 run first to measure time.

import csv
import os
import sys
import time
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False
    print("Warning: psutil not installed, RAM will be 0. Run: pip install psutil", flush=True)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add src to path for Peanut
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from optimizers.peanut import Peanut

class SmallMLP(nn.Module):
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

def run_one(seed, optimizer_name):
    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='data', train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root='data', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = SmallMLP()
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    else:
        opt = Peanut(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
    start_time = time.time()
    start_ram = process.memory_info().rss / (1024*1024) if HAS_PSUTIL else 0  # MB
    peak_ram = start_ram
    total_loss = 0.0
    total_batches = 0

    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_batches += 1
        # Check RAM each epoch
        cur_ram = process.memory_info().rss / (1024*1024) if HAS_PSUTIL else 0
        print(f"  {optimizer_name} seed {seed} epoch {epoch+1}/15 done, RAM {cur_ram:.0f} MB", flush=True)
        if cur_ram > peak_ram:
            peak_ram = cur_ram

    end_time = time.time()
    total_time = end_time - start_time
    avg_loss = total_loss / total_batches

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    print(f"Result: {optimizer_name} seed {seed} Acc {acc:.4f} Loss {avg_loss:.4f} Time {total_time:.1f}s PeakRAM {peak_ram:.0f} MB", flush=True)
    return [optimizer_name, seed, acc, avg_loss, total_time, peak_ram]

def main():
    os.makedirs('results', exist_ok=True)
    # For today, run for full 9 runs.
    # Today: 9 runs
    jobs = [(opt, seed) for opt in ['Adam', 'AdamW', 'Peanut'] for seed in [0, 1, 42]]
    # Full 9 runs, use this later:
    # jobs = [(opt, seed) for opt in ['Adam', 'AdamW', 'Peanut'] for seed in [0, 1, 42]]
    rows = []
    overall_start = time.time()
    for opt_name, seed in jobs:
        row = run_one(seed, opt_name)
        rows.append(row)
    overall_time = time.time() - overall_start
    print(f"Done 1 run in {overall_time:.1f}s. Estimated 9 runs: {overall_time*9:.1f}s = {overall_time*9/60:.1f} mins", flush=True)
    # Append to CSV, do not overwrite
    file_exists = os.path.exists('results/mnist_baseline.csv')
    with open('results/mnist_baseline.csv', 'a', newline='') as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(['optimizer', 'seed', 'accuracy', 'loss', 'time_sec', 'peak_ram_mb'])
        w.writerows(rows)
    print("Saved to results/mnist_baseline.csv", flush=True)

if __name__ == '__main__':
    main()