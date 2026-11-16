# fashion_baseline.py - Same pipeline as mnist_baseline.py, only dataset changed
# Fashion-MNIST, MLP, 10 epochs, 3 seeds (0,1,42), Adam vs AdamW vs Peanut
# Progress: start/stop per run only, not verbose per batch
# Output: results/fashion_baseline.csv (local only, not committed)

import csv
import os
import time
import torch
import torch.nn as nn
import tracemalloc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, 'src')
from optimizers.peanut import Peanut

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

def run_one(optimizer_name, seed, run_idx, total_runs, start_all):
    print(f"\n[{run_idx}/{total_runs}] START {optimizer_name} seed {seed} ...", flush=True)
    run_start = time.time()
    torch.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root='data', train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST(root='data', train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = TinyMLP()
    if optimizer_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    else:
        opt = Peanut(model.parameters(), lr=0.01)

    loss_fn = nn.CrossEntropyLoss()
    tracemalloc.start()
    start = time.time()
    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            total_loss += loss_fn(out, y).item() * len(y)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    loss = total_loss / total
    run_time = time.time() - run_start
    done = run_idx
    remaining = total_runs - done
    avg_per_run = (time.time() - start_all) / done
    eta_min = avg_per_run * remaining / 60
    print(f"[{run_idx}/{total_runs}] DONE {optimizer_name} seed {seed} | Acc {acc:.4f} Loss {loss:.4f} Time {elapsed:.1f}s PeakRAM {peak_mb:.0f} MB | Elapsed {run_time:.0f}s, ETA {eta_min:.1f} min for {remaining} runs left", flush=True)
    return acc, loss, elapsed, peak_mb

def main():
    os.makedirs('results', exist_ok=True)
    optimizers = ['Adam', 'AdamW', 'Peanut']
    seeds = [0, 1, 42]
    total_runs = len(optimizers) * len(seeds)
    print(f"=== Fashion-MNIST Baseline START ===", flush=True)
    print(f"Optimizers: {optimizers}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Total runs: {total_runs} (10 epochs each)", flush=True)
    print(f"Results will be saved to results/fashion_baseline.csv", flush=True)
    start_all = time.time()
    rows = []
    run_idx = 0
    for opt_name in optimizers:
        print(f"\n--- Starting optimizer: {opt_name} ---", flush=True)
        for seed in seeds:
            run_idx += 1
            acc, loss, t, ram = run_one(opt_name, seed, run_idx, total_runs, start_all)
            rows.append([opt_name, seed, acc, loss, t, ram])
        print(f"--- Finished optimizer: {opt_name} ---", flush=True)

    with open('results/fashion_baseline.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['optimizer', 'seed', 'accuracy', 'loss', 'time_sec', 'peak_ram_mb'])
        w.writerows(rows)
    total_time = time.time() - start_all
    print(f"\n=== Fashion-MNIST Baseline DONE ===", flush=True)
    print(f"Total time: {total_time/60:.1f} mins for {total_runs} runs", flush=True)
    print(f"Saved to results/fashion_baseline.csv", flush=True)
    # Summary
    for opt_name in optimizers:
        accs = [r[2] for r in rows if r[0] == opt_name]
        mean = sum(accs) / len(accs)
        print(f"Summary {opt_name}: Mean Acc {mean:.4f} | Runs: {accs}", flush=True)

if __name__ == '__main__':
    main()