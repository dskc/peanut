# experiments/cnn_exp02.py - Experiment 02, CNN, 20 epochs, 3 seeds
# Public validation: Adam vs Peanut v2, long training
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

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def train_one(name, seed=0, epochs=20, lr=0.01):
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=g)
    model = TinyCNN()
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
        print(f"{name} seed={seed} epoch={epoch+1}/{epochs} acc={acc:.4f}", flush=True)
    print(f"{name} seed={seed} DONE final={accs[-1]:.4f} max={max(accs):.4f}", flush=True)
    return accs

if __name__ == "__main__":
    seeds = [0, 1, 42]
    all_results = {}
    for n in ["adam", "peanut_v2"]:
        finals = []
        maxs = []
        for s in seeds:
            accs = train_one(n, seed=s, epochs=20, lr=0.01)
            finals.append(accs[-1])
            maxs.append(max(accs))
        all_results[n] = (finals, maxs)
        print(f"{n} finals: {finals} mean={np.mean(finals):.4f} std={np.std(finals):.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/cnn_exp02.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["optimizer", "seed0", "seed1", "seed42", "mean", "std", "mean_max"])
        for n in ["adam", "peanut_v2"]:
            finals, maxs = all_results[n]
            w.writerow([n, finals[0], finals[1], finals[2], np.mean(finals), np.std(finals), np.mean(maxs)])
    print("Saved: results/cnn_exp02.csv", flush=True)