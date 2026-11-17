# Peanut - Roadmap / Direction

Last updated: 17 Nov 2026
Status: Direction, not results. This is where Peanut is going, and how we will prove it.
Current phase: Phase 1 - Prove it on 2 datasets, close the 4 unproven gaps

## Principle

CPU-only is a design decision, not a shortcoming.

Peanut was designed for CPU. Validated on CPU. Stable on CPU.
If it works on CPU, it works anywhere. If it needs a GPU to be stable, it is not stable.
CPU forces honesty: no big batch to hide instability, no fast GPU to hide slow optimizer.
CPU is what most people have. A student with a laptop can run Peanut.

We do not wait for GPUs. We design for the machine we have.

## What is proven until now

- MNIST, MLP, 3 seeds: Adam 0.9787, AdamW 0.9793, Peanut 0.9806. Peanut wins.
- Fashion-MNIST, MLP, 3 seeds: Adam 0.8605, AdamW 0.8552, Peanut 0.8611. Peanut wins.
- Peanut is stable, does not collapse at high LR.
- Peanut time and RAM are same or better on CPU: MNIST, Adam 218.8s, AdamW 227.4s, Peanut 203.9s, RAM all ~383 MB.
- Ablation: Neither damp alone nor inflated v alone beats Adam. Both together beat Adam at high LR.
- Zero-m test: Peanut shipped 0.8579 vs Peanut with Adam m 0.8577, mean difference 0.0002. Zero-m carries nothing for mean, hurts worst seed.

## What remains unproven - Our target is beyond these

1. **Consistent superiority across datasets**
   Proven on MNIST + Fashion-MNIST only. Not on CIFAR-10, regression, NLP.

2. **Superiority over a properly tuned AdamW**
   Until now, AdamW was tested at same LR as Peanut, not at AdamW's own best LR. We must tune AdamW separately.

3. **Robustness across architectures**
   Proven on MLP and small CNN only. Not on ResNet, Transformer, larger networks.

4. **Statistically significant improvement**
   Proven with 3 seeds, mean difference 0.13% - 0.19%. With 3 seeds, that could be luck. Need 5 seeds, mean, std, worst, and check.

Our target is to close these 4 gaps, not to add more features first.

## Primary focus

Use the same datasets, multiple random seeds, and properly tuned learning rates, to confidently say Peanut is better.

Rules for every experiment:
- Same dataset for all optimizers
- Same model for all optimizers
- Same split for all optimizers
- Each optimizer gets its own best LR, not Peanut's LR forced on others
- Multiple seeds, not 1 seed

## 5 Questions we must answer every time

1. Convergence speed: Does Peanut reach the same accuracy in fewer training steps?
2. Generalization: Does it perform better on unseen data?
3. Stability: Does it avoid exploding gradients and unstable updates?
4. Efficiency: Does it consume less CPU memory and training time? And if possible, check on GPU too, for larger community, even though Peanut is CPU-based.
5. Scalability: Does it work well with larger neural networks and Transformers?

## Metrics - In priority order

| Metric | Why it matters |
| --- | --- |
| Convergence rate | How quickly does it minimize loss? |
| CPU execution time | How quickly does each optimization step execute? |
| CPU memory usage | How much RAM does it consume? |
| Generalization | Does it perform well on unseen data? |
| Numerical stability | Does training remain stable? |
| Hyperparameter sensitivity | How much tuning does it require? |
| Reproducibility | Does it produce consistent results? |

Every experiment must measure: Validation accuracy, Training loss, Convergence speed, CPU memory consumption, Training time, Performance across 5 random seeds.

## Baselines - Why Peanut competes against each

For a fair CPU-based benchmark, we use these competitors. All can be evaluated on CPU. There is no requirement to use GPU.

| Competitor | Why include it? |
| --- | --- |
| Adam | Original baseline |
| AdamW | Strong modern baseline |
| SGD + Momentum | Generalization comparison |
| RMSProp | Adaptive learning-rate comparison |
| Adagrad | Adaptive gradient comparison |
| Lion | Alternative momentum-based optimizer |

Primary test: Peanut vs Adam vs AdamW - must win.
Full test: Peanut vs all 6 - to prove *why* it wins, not just *that* it wins.

## Seeds

Use 5 prime seeds for paper-level tests: 1, 5, 7, 11, 13
Use 3 seeds for quick tests: 0, 1, 42
Report Mean, Std, Worst, Best for all seeds, not just Mean.

## Datasets - In order

- MNIST: Done. Basic classification and convergence testing.
- Fashion-MNIST: Done, now. More challenging image classification.
- CIFAR-10: Only after MNIST + Fashion-MNIST justify it. Small CNN first, not ResNet-18.
- California Housing, IMDB, ResNet-18: Later, only if justified.

One reusable training pipeline, change only dataset and model configuration. No separate codebase per dataset.

## Framework - One pipeline

Build one reusable experimental framework with:
- Common dataset loading
- Common model definitions
- Common training loop
- Common evaluation metrics
- Fixed random seeds
- Automatic CSV result logging
- CPU execution-time measurement
- Peak memory measurement

This prevents each optimizer from having a different experimental implementation.

## Features - In order, incremental

Do not implement all at once. One feature at a time, ablation after each, keep only what helps.

1. Adaptive momentum - Implement first
2. Adaptive gradient smoothing - Implement second
3. Adaptive learning-rate control - Later
4. Adaptive gradient clipping - Later
5. AdamW-style weight decay - Later

Do NOT implement in first version:
- FP16 optimizer states - Postpone
- Factored optimizer states - Postpone
- Curvature approximation - Postpone
- DeepSpeed CPUAdam - Optional, only if environment supports it

## Milestones - Designed for ordinary CPU laptops

### Milestone 1 - Baseline - DONE
- Reproduce Peanut v2.3
- Compare against Adam and AdamW
- MNIST with small MLP, 3 seeds, 10-20 epochs
- Status: MNIST done, Fashion-MNIST done

### Milestone 2 - Peanut v3: Adaptive improvements - NEXT
- Adaptive momentum
- Adaptive gradient smoothing
- Test each feature independently
- Retain only measurable improvements

### Milestone 3 - Peanut v3.1: Stability
- Adaptive gradient clipping
- Adaptive learning-rate control
- AdamW-style weight decay
- Fashion-MNIST validation

### Milestone 4 - Peanut v4: CPU efficiency
- Reduce temporary tensor allocations
- Optimize vectorized operations
- Reduce optimizer-state memory overhead
- Benchmark actual CPU training time

### Milestone 5 - Peanut v4: Generalization
- Add CIFAR-10 with small CNN first
- Expand seeds to 5
- Test larger architectures only if justified
- Evaluate across datasets

## Starting experiment - 9 runs, not 60

| Parameter | Initial setting |
| --- | --- |
| Dataset | MNIST |
| Architecture | Small MLP |
| Optimizers | Adam, AdamW, Peanut |
| Random seeds | 3 |
| Epochs | 15 |
| Batch size | 64 |
| Execution | CPU only |
| Metrics | Accuracy, loss, time, RAM |

Total: 9 training runs, not 60. If results are promising, increase experiment size.

## Links

- Repo: https://github.com/dskc/peanut
- README: Results that are done
- TRACKER.md: What we are doing now
- EVOLUTION_LOG.md: This file, where we are going