# cifar10_exp01.py - Phase 1, CIFAR-10 Validation
# TinyCNN, CIFAR-10, batch 128, 50 epochs, 3 seeds (0,1,42)
# Optimizers: AdamW vs Peanut v2.2, LR 0.003 and 0.01
# Output: results/cifar10_exp01.csv
# Run on Colab Free, not laptop. Save CSV to Drive after each seed.

import csv
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add src to path so we can import Peanut
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from optimizers.peanut import Peanut

# --- TinyCNN for CIFAR-10, RGB 32x32 ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_loaders(seed):
    torch.manual_seed(seed)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = datasets.CIFAR10(root='data', train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root='data', train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, num_workers=2)
    return train_loader, test_loader

def run_one(seed, optimizer_name, lr):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Start: {optimizer_name} LR {lr} seed {seed} on {device}", flush=True)
    train_loader, test_loader = get_loaders(seed)
    torch.manual_seed(seed)
    model = TinyCNN().to(device)
    if optimizer_name == 'AdamW':
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = Peanut(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        if (epoch+1) % 10 == 0:
            print(f"  {optimizer_name} LR {lr} seed {seed} epoch {epoch+1}/50 done", flush=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    acc = correct / total
    print(f"Result: {optimizer_name} LR {lr} seed {seed}: {acc:.4f}", flush=True)
    return acc

def main():
    os.makedirs('results', exist_ok=True)
    rows = []
    for optimizer_name in ['AdamW', 'Peanut']:
        for lr in [0.003, 0.01]:
            accs = []
            for seed in [0, 1, 42]:
                acc = run_one(seed, optimizer_name, lr)
                accs.append(acc)
            mean = sum(accs) / len(accs)
            worst = min(accs)
            print(f"FINAL: {optimizer_name} LR {lr} MEAN {mean:.4f} WORST {worst:.4f}")
            rows.append([optimizer_name, lr, mean, worst] + accs)
            # Save after each LR block, so Colab disconnect does not lose all
            with open('results/cifar10_exp01.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['optimizer', 'lr', 'mean', 'worst', 'seed0', 'seed1', 'seed42'])
                w.writerows(rows)
            print("Saved to results/cifar10_exp01.csv", flush=True)

if __name__ == '__main__':
    main()