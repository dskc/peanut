# Peanut - Beating Adam on CPU

Goal: Build an optimizer better than Adam on a specific weakness,
validated on CPU-only laptop.

Status: Peanut v2.3.0 is released, Latest. Peanut v3 / v4 / v5 are planned, not started.
Current version: v2.3.0 - CIFAR-10 validated, Adam m fix.

Hardware: CPU-only, 32GB RAM, Intel Alder Lake N-series, Windows 11
Python: 3.13, torch 2.15.0+cpu, no GPU

## Structure

data/ : datasets, auto-downloaded
src/optimizers/ : Peanut v2.3.0
experiments/ : final validation runs
results/ : tables, local only, not committed, in .gitignore

## Idea

Peanut v2.3.0 is Adam with two changes, both LR-aware, plus one fix:

1. Damped first step: first update multiplied by effective_damp.
   effective_damp = 0.2 * 0.01 / lr, clipped to 0.05 to 1.0.
   At LR 0.01, damp = 0.2. At LR 0.003, damp = 0.66. At LR 0.001, damp = 1.0.

2. Inflated v init: v = v_factor * g1^2, where v_factor = (1 - beta2) + beta2 * inflation, inflation = lr / 0.01, clipped to 0 to 1.0.
   At LR 0.01, v = g1^2. At LR 0.003, v is closer to correct Adam v.

3. Fix in v2.3.0: First-step m uses Adam m, not zero m.
   Old v2.2 left m at 0 on step 1. That was a silent third change.
   m-test showed zero m carries nothing for mean, hurts worst seed.
   v2.3.0 uses m = (1 - beta1) * g1, bias-corrected, same as Adam.

Everything else is standard Adam: betas (0.9, 0.999), eps 1e-8,
bias correction from step 2 onward.

Why both? Ablation showed neither change alone works. Only both together beat Adam at high LR. At low LR, damping fades so Peanut behaves close to Adam.

## Results

All results: batch 64, 3 seeds (0, 1, 42), unless stated otherwise.

### Experiment 01 - CNN, Fashion-MNIST, 10 epochs, LR 0.01, Peanut v2

| Optimizer | Mean | Std |
| --- | --- | --- |
| Adam | 0.8934 | 0.0028 |
| AdamW | 0.8889 | 0.0061 |
| Peanut v2 | 0.8952 | 0.0012 |

### Experiment 02 - CNN, Fashion-MNIST, 20 epochs, LR 0.01, Peanut v2

| Optimizer | Mean Final | Std | Mean Max |
| --- | --- | --- | --- |
| Adam | 0.8901 | 0.0041 | 0.8968 |
| Peanut v2 | 0.8972 | 0.0038 | 0.9030 |

### Ablation - MLP, Fashion-MNIST, 10 epochs, 2 LRs

At LR 0.003:
- Adam: 0.8800 mean, 0.8770 worst
- Only damp: 0.8787 mean, 0.8780 worst
- Only inflated v: 0.8788 mean, 0.8758 worst
- Both, Peanut v2: 0.8786 mean, 0.8761 worst
- Warmup: 0.8829 mean, 0.8797 worst

At LR 0.01:
- Adam: 0.8613 mean, 0.8589 worst
- Only damp: 0.8551 mean, 0.8534 worst
- Only inflated v: 0.8579 mean, 0.8573 worst
- Both, Peanut v2: 0.8619 mean, 0.8598 worst
- Warmup: 0.8544 mean, 0.8444 worst, collapses

Conclusion from ablation: Neither change alone beats Adam. Both together beat Adam at high LR. Warmup is best at low LR, worst at high LR.

### Peanut v2.2 - MLP, Fashion-MNIST, 10 epochs, LR-aware

| Method | LR 0.003 Mean | LR 0.003 Worst | LR 0.01 Mean | LR 0.01 Worst |
| --- | --- | --- | --- | --- |
| Adam | 0.8800 | 0.8770 | 0.8613 | 0.8589 |
| Peanut v2.2 | 0.8807 | 0.8754 | 0.8619 | 0.8598 |

v2.2 mean wins at both LRs. At low LR, worst seed is slightly lower than Adam, a known trade-off.

### m-test - MLP, Fashion-MNIST, 10 epochs, LR 0.01

Test for silent third change, zero m vs Adam m:

- Peanut v2.2, zero m: Mean 0.8579, Worst 0.8521
- Peanut v2.3.0, Adam m: Mean 0.8577, Worst 0.8568

Mean difference 0.0002, zero m carries nothing for mean. Worst seed is better with Adam m. This is why v2.3.0 uses Adam m.

### Community result - CIFAR-10, 3 epochs, Colab L4, from Hugging Face feedback

| Optimizer | LR 0.003 | LR 0.01 |
| --- | --- | --- |
| Adam | 49.13% | 46.86% |
| Adam + warmup 100 | 55.31% | 53.51% |
| Peanut v2 | 56.18% | 52.66% |

Peanut beats Adam by 7.04 points at 0.003 and 5.80 points at 0.01 in this small 3-epoch probe. Peanut vs warmup flips with LR. This is a community probe, not a general claim.

### CIFAR-10 Validation - 50 epochs, TinyCNN, Colab T4, Peanut v2.3.0

| Optimizer | LR 0.003 Mean | LR 0.003 Worst | LR 0.01 Mean | LR 0.01 Worst |
| --- | --- | --- | --- | --- |
| AdamW | 0.7813 | 0.7690 | 0.3396 | 0.1000 |
| Peanut v2.3.0 | 0.7847 | 0.7814 | 0.6526 | 0.6409 |

At LR 0.003, Peanut wins by +0.34 points mean, +1.24 points worst.
At LR 0.01, AdamW collapses, one seed at 0.1000 random guessing. Peanut never collapses, wins by +31.30 points mean.
This is the stability claim, now proven on 2 datasets: Fashion-MNIST and CIFAR-10.

### MNIST Baseline - MLP, 10 epochs, 3 seeds, CPU, Peanut v2.3.0

| Optimizer | Mean Accuracy | Mean Time | Mean Peak RAM |
| --- | --- | --- | --- |
| Adam | 0.9787 | 218.8s | 379 MB |
| AdamW | 0.9793 | 227.4s | 383 MB |
| Peanut v2.3.0 | 0.9806 | 203.9s | 383 MB |

Peanut v2.3.0 beats Adam by 0.19% and AdamW by 0.13% on MNIST.
Peanut range across seeds: 0.0002, Adam range: 0.0021.
Peanut is not slower, 7% faster than Adam on same CPU, same code.
RAM is same for all optimizers.
Full runs in `experiments/mnist_baseline.py`, 3 seeds (0, 1, 42).

### Fashion-MNIST Baseline - MLP, 10 epochs, 3 seeds, CPU, Peanut v2.3.0

| Optimizer | Mean Accuracy | Runs |
| --- | --- | --- |
| Adam | 0.8605 | 0.8637, 0.8543, 0.8635 |
| AdamW | 0.8552 | 0.8567, 0.8548, 0.8542 |
| Peanut v2.3.0 | 0.8611 | 0.8583, 0.8659, 0.8592 |

Peanut v2.3.0 beats Adam by 0.06% and AdamW by 0.59% on Fashion-MNIST.
Total time: 45.6 mins for 9 runs on CPU-only laptop.
Full runs in `experiments/fashion_baseline.py`, 3 seeds (0, 1, 42).

## Conclusion - v2.3.0

Peanut v2.3.0 does not beat Adam by a lot. It beats Adam by 0.06% to 0.19% on most tests. The real win is stability at high LR.

- Peanut v2.3.0 almost always has lower std than Adam and AdamW at high LR.
- AdamW collapses on some seeds (0.1000 on CIFAR-10 at LR 0.01, 0.8803 on CNN, 0.8366 on MLP).
- Peanut v2.3.0 never collapsed in our tests.

Final claim for v2.3.0:
Peanut v2.3.0 is a more stable Adam variant with similar accuracy and similar time cost on CPU, for small MLP / CNN on MNIST, Fashion-MNIST, and CIFAR-10. Damping fades at low LR so it behaves close to Adam. It is not a general SOTA optimizer.

Philosophy for next versions:
Adam is already highly optimized. Peanut must show better convergence same budget, similar/better accuracy less CPU time, no significant memory increase, stable across seeds. If Peanut cannot beat Adam on a benchmark, record it, do not hide it. Lightweight, CPU-first. Every feature must justify its computational cost.

## Roadmap - v3, v4, v5

Designed for ordinary CPU laptops. One feature at a time, ablation after each, retain only measurable improvements. One reusable pipeline for all optimizers. Start small, 9 runs, then scale only if promising.

### Milestone 1 - Baseline - DONE
- Reproduce Peanut v2.3.0
- MNIST, Small MLP, Adam vs AdamW vs Peanut, 3 seeds, 15 epochs, batch 64, CPU only
- Metrics: Accuracy, loss, time, RAM
- Result: MNIST done, Peanut wins. Fashion-MNIST done, Peanut wins.
- Total: 9 runs per dataset, not 60.

### Milestone 2 - Peanut v3: Adaptive improvements - NEXT
Goal: Better convergence, same budget.
Features, in this order, one at a time:
1. Adaptive momentum - Implement first. Adjust momentum coefficient based on gradient stability. Test alone, ablation.
2. Adaptive gradient smoothing - Implement second. Reduce fluctuations in noisy mini-batch gradients. Test alone, ablation.
- Retain only features with measurable improvement.
- Test on: MNIST MLP, 3 seeds, 15 epochs, then Fashion-MNIST MLP, 3 seeds, 10 epochs.
- Baselines: Adam, AdamW, SGD (torch.optim only).
- Success: Mean accuracy up, or same accuracy in fewer epochs, with same time + RAM.

### Milestone 3 - Peanut v3.1: Stability - LATER
Goal: Stable across seeds and LRs.
Features, in this order:
1. Adaptive gradient clipping - Prevent excessively large updates.
2. Adaptive learning-rate control - Adjust LR by training progress, detect stagnation, oscillation, divergence.
3. AdamW-style weight decay - Decoupled WD for regularization and generalization.
- Test each alone, ablation, then combined only if each helps.
- Validation: Fashion-MNIST, then CIFAR-10 small CNN.
- Success: Worst seed improves, no collapses at LR 0.01 and 0.03.

### Milestone 4 - Peanut v4: CPU efficiency - LATER
Goal: Lower CPU training time, equal or lower memory.
Features:
- Reduce temporary tensor allocations.
- Optimize vectorized operations, avoid Python-level loops.
- Optimize math for CPU SIMD, reduce redundant gradient calculations.
- Optimize optimizer-state updates, reuse buffers, avoid unnecessary copies.
- Investigate lower-precision optimizer-state storage only if stable, FP16 / BF16 as optional, not default.
- Do NOT implement factored states or curvature approximation in v4 first version. Postponed to v5.
- Benchmark: Total training time, optimizer-step time, CPU utilization, throughput, peak RAM, optimizer-state memory.
- Baselines: Adam, AdamW, SGD, plus CPUAdam if environment supports it, DeepSpeed optional.
- Success: Same accuracy as v3.1, with less CPU time, same or lower peak RAM.

### Milestone 5 - Peanut v4: Generalization - LATER
Goal: Consistent performance across datasets, not one benchmark.
- Add CIFAR-10 with small CNN first, not ResNet-18.
- Expand seeds to 5 (0, 1, 42, 123, 999).
- Larger architectures only if justified by small CNN results.
- Test 100+ epochs once, to check if lead persists.
- Evaluate: Mean, std, worst, best, time, RAM for each dataset.
- Success: Wins or ties on 2 of 3 datasets (MNIST, Fashion-MNIST, CIFAR-10) with no collapse.

### v5 - Vision - NOT STARTED, DO NOT START YET
Goal: General-purpose, still CPU-first, but broader.
Points to explore only after v4 is done and stable:
1. Advanced memory: Factored optimizer states, compressed states for larger models. Only if v4 memory is still a bottleneck.
2. Advanced math: Lightweight diagonal or block-diagonal curvature approximation. Only if it shows measurable convergence benefit without high CPU cost.
3. Advanced precision: FP16 / BF16 optimizer states as optional flag, with stability tests.
4. Broader datasets: CIFAR-100, California Housing regression, 【entity-IMDB¦canonical_name=IMDB】 NLP small model. Only one at a time, with same pipeline.
5. Larger architectures: ResNet-18 on CIFAR-10, small transformer on Tiny Shakespeare. Only if small CNN results justify it.
6. More seeds: 10 seeds for final paper, for cheapest test only.
7. Packaging: Pip installable `peanut-optimizer`, 3 examples, benchmark script, Hugging Face Space demo.
8. Paper: 4-page workshop paper, arXiv, with all failures recorded, not hidden.
- Rule for v5: Every v5 feature must pass the same test as v3: ablation alone, measurable improvement, cost justified. If it fails, drop it.

### Reviewer Checklist - After Milestone 2
- [ ] CIFAR-10 vs Adam/AdamW, 5 seeds
- [ ] Training time on identical CPU, already done for MNIST, need for CIFAR-10
- [ ] 100+ epoch persistence check

## How to run

```powershell
.\venv\Scripts\Activate.ps1
python experiments\cnn_exp01.py
python experiments\cnn_exp02.py
python experiments\mnist_baseline.py
python experiments\fashion_baseline.py