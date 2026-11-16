# Peanut - Revised Plan - Laptop Only - Active - 16 Nov 2026
Current phase: Milestone 1 - Baseline, MNIST 9-run experiment
Current status: Planning done, code NOT started, waiting to start Milestone 1
Old plan saved in OLD_PLAN.md, do not execute old plan.

# Peanut - Tracker / TODO
Current phase: Phase 1 - CIFAR-10 validation
Current status: Peanut v2.2.0 shipped, MNIST baseline done, Peanut wins

## Phase 0 - Done - Lock v2.2
- [x] Peanut v2.2 shipped, Release v2.2.0
- [x] README fixed, no merge markers, tail present
- [x] LinkedIn draft ready

## Phase 1 - NOW - CIFAR-10 + Reviewer Checklist
Goal: Prove Peanut on 2nd dataset, answer reviewer 4 points

### MNIST Baseline - Done 16 Nov 2026
- [x] MNIST, 9 runs, 3 seeds (0,1,42)
- [x] Adam mean 0.9787, AdamW mean 0.9793, Peanut mean 0.9806
- [x] Peanut wins by 0.19% over Adam, 0.13% over AdamW
- [x] Time: Adam 218.8s, AdamW 227.4s, Peanut 203.9s - Peanut not slower
- [x] RAM: All ~383 MB, same
- [x] File: results/mnist_baseline.csv

### R1 - CIFAR-10 vs Adam and AdamW - TODO
- [ ] CIFAR-10, TinyCNN, 50 epochs, batch 128, LR 0.003 and 0.01
- [ ] Optimizers: Adam vs AdamW vs Peanut v2.2
- [ ] 5 seeds: 0, 1, 42, 123, 999

### R2 - Training time on identical CPU - TODO
- [ ] Already DONE for MNIST, need same for CIFAR-10
- [ ] Log seconds per epoch, total seconds, for all optimizers

### R3 - 5-10 seeds - TODO
- [ ] Use 5 seeds for CIFAR-10
- [ ] Report Mean, Std, Worst

### R4 - 100+ epochs - TODO
- [ ] CIFAR-10, 100 epochs, AdamW vs Peanut v2.2, 3 seeds first

## Phase 2 - TODO - Stronger Baselines
- [ ] Add Lion, test transformer, Tiny Shakespeare

## Phase 3 - TODO - Make it Usable
- [ ] Pip installable, 3 examples

## Phase 4 - TODO - Paper
- [ ] 4-page paper, Hugging Face blog

## V4 Vision - Do Not Start Yet
- [ ] Decoupled weight decay alone, as v2.3
- [ ] Adaptive clipping alone, as v2.4
- [ ] Do NOT start until Phase 1 is [x]

## Philosophy
Adam is already highly optimized and extensively researched.
Peanut must show: better convergence same budget, similar/better accuracy less CPU time, no significant memory increase, stable across seeds.
If Peanut cannot beat Adam on a benchmark, record it, do not hide it.
Lightweight, CPU-first. Every feature must justify its computational cost.
First milestone: reproducible improvement on MNIST + Fashion-MNIST on ordinary laptop.

## Feature Order - One at a time, ablation after each
- [ ] 1. Adaptive momentum - Implement first
- [ ] 2. Adaptive gradient smoothing - Implement second
- [ ] 3. Adaptive learning-rate control - Later
- [ ] 4. Adaptive gradient clipping - Later
- [ ] 5. AdamW-style weight decay - Later
- [ ] Postponed: FP16 states, Factored states, Curvature approximation
- [ ] Optional: DeepSpeed CPUAdam, only if environment supports it

## Datasets - Only 2 to start
- [ ] MNIST - Basic classification + convergence
- [ ] Fashion-MNIST - More challenging image classification
- [ ] CIFAR-10 - Only after promise on MNIST + Fashion-MNIST
- [ ] Dropped for now: 【entity-IMDB¦canonical_name=IMDB】, California Housing, ResNet-18

## Baselines - torch.optim only
- Adam, AdamW, SGD
- DeepSpeed CPUAdam: Optional later

## Framework - One reusable pipeline
- [ ] Common dataset loading
- [ ] Common model definitions
- [ ] Common training loop
- [ ] Common evaluation metrics
- [ ] Fixed random seeds
- [ ] Automatic CSV logging
- [ ] CPU execution-time measurement
- [ ] Peak memory measurement
- Purpose: Prevent each optimizer from having different experimental implementation

## Milestone 1 - Baseline - NOW
- [ ] Reproduce Peanut v2.3 (v2.2 + Adam m fix)
- [ ] Experiment: MNIST, Small MLP, Adam vs AdamW vs Peanut, 3 seeds, 15 epochs, batch 64, CPU only
- [ ] Metrics: Accuracy, loss, time, RAM
- [ ] Total: 9 runs (3 optimizers x 3 seeds). If promising, increase size. If not, record failure.

## Milestone 2 - Peanut v3: Adaptive improvements
- [ ] Adaptive momentum, test alone, ablation
- [ ] Adaptive gradient smoothing, test alone, ablation
- [ ] Retain only measurable improvements

## Milestone 3 - Peanut v3.1: Stability
- [ ] Adaptive gradient clipping
- [ ] Adaptive learning-rate control
- [ ] AdamW-style weight decay
- [ ] Fashion-MNIST validation

## Milestone 4 - Peanut v4: CPU efficiency
- [ ] Reduce temporary tensor allocations
- [ ] Optimize vectorized operations
- [ ] Reduce optimizer-state memory overhead
- [ ] Benchmark actual CPU training time

## Milestone 5 - Peanut v4: Generalization
- [ ] Add CIFAR-10 with small CNN first
- [ ] Expand seeds to 5
- [ ] Larger architectures only if justified
- [ ] Evaluate across datasets

## Reviewer Checklist - After Milestone 1, not before
- [ ] CIFAR-10 vs Adam/AdamW, 5 seeds
- [ ] Training time on identical CPU
- [ ] 100+ epoch persistence check

## Starting Experiment - Exact settings
Dataset: MNIST
Architecture: Small MLP
Optimizers: Adam, AdamW, Peanut
Seeds: 3
Epochs: 15
Batch size: 64
Execution: CPU only
Metrics: Accuracy, loss, time, RAM
Total runs: 9

## Rules
1. One feature at a time, ablation after each
2. One pipeline for all optimizers
3. 3 seeds now, 5 seeds later, 10 seeds only for cheapest test
4. Record failures, do not hide
5. Doing column has only 1 card
6. If experiment >2h on laptop, use Colab/Kaggle, not laptop
7. Update TRACKER.md only on Friday, Kanban handles daily tracking

## Links
- Repo: https://github.com/dskc/peanut
- Project board: https://github.com/dskc/peanut/projects - 5 cards: M1, M2, M3, M4, M5
- Old plan: OLD_PLAN.md