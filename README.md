# Peanut - Beating Adam on CPU

Goal: Build an optimizer better than Adam on a specific weakness,
validated on CPU-only laptop.

Status: Completed. Peanut v2.2 is the final version, LR-aware.

Hardware: CPU-only, 32GB RAM, Intel Alder Lake N-series, Windows 11
Python: 3.13, torch 2.15.0+cpu, no GPU

## Structure

data/ : datasets, auto-downloaded
src/optimizers/ : Peanut v2.2
experiments/ : final validation runs
results/ : tables

## Idea

Peanut v2.2 is Adam with two changes, both LR-aware:

1. Damped first step: first update multiplied by effective_damp.
   effective_damp = 0.2 * 0.01 / lr, clipped to 0.05 to 1.0.
   At LR 0.01, damp = 0.2. At LR 0.003, damp = 0.66. At LR 0.001, damp = 1.0.

2. Inflated v init: v = v_factor * g1^2, where v_factor = (1 - beta2) + beta2 * inflation, inflation = lr / 0.01, clipped to 0 to 1.0.
   At LR 0.01, v = g1^2. At LR 0.003, v is closer to correct Adam v.

Everything else is standard Adam: betas (0.9, 0.999), eps 1e-8,
bias correction from step 2 onward.

Why both? Ablation showed neither change alone works. Only both together beat Adam at high LR. At low LR, damping fades so Peanut behaves close to Adam.

## Results

All results: Fashion-MNIST, batch 64, 3 seeds (0, 1, 42).

<<<<<<< HEAD
### Experiment 01 - CNN, 10 epochs
=======
### Experiment 01 - CNN, 10 epochs, LR 0.01, Peanut v2
>>>>>>> 740cb02 (Peanut v2.2 - LR-aware damping, ablation + community CIFR-10 results)

| Optimizer | Mean | Std |
| --- | --- | --- |
| Adam | 0.8934 | 0.0028 |
| AdamW | 0.8889 | 0.0061 |
| Peanut v2 | 0.8952 | 0.0012 |

<<<<<<< HEAD
### Experiment 02 - CNN, 20 epochs
=======
### Experiment 02 - CNN, 20 epochs, LR 0.01, Peanut v2
>>>>>>> 740cb02 (Peanut v2.2 - LR-aware damping, ablation + community CIFR-10 results)

| Optimizer | Mean Final | Std | Mean Max |
| --- | --- | --- | --- |
| Adam | 0.8901 | 0.0041 | 0.8968 |
| Peanut v2 | 0.8972 | 0.0038 | 0.9030 |

<<<<<<< HEAD
Note: Experiment 02 was run to check long training. Peanut v2 stays stable on long training.
=======
### Ablation - MLP, 10 epochs, 2 LRs

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

### Peanut v2.2 - MLP, 10 epochs, LR-aware

| Method | LR 0.003 Mean | LR 0.003 Worst | LR 0.01 Mean | LR 0.01 Worst |
| --- | --- | --- | --- | --- |
| Adam | 0.8800 | 0.8770 | 0.8613 | 0.8589 |
| Peanut v2.2 | 0.8807 | 0.8754 | 0.8619 | 0.8598 |

v2.2 mean wins at both LRs. At low LR, worst seed is slightly lower than Adam, a known trade-off.

### Community result - CIFAR-10, 3 epochs, Colab L4, from Hugging Face feedback

| Optimizer | LR 0.003 | LR 0.01 |
| --- | --- | --- |
| Adam | 49.13% | 46.86% |
| Adam + warmup 100 | 55.31% | 53.51% |
| Peanut v2 | 56.18% | 52.66% |

Peanut beats Adam by 7.04 points at 0.003 and 5.80 points at 0.01 in this small 3-epoch probe. Peanut vs warmup flips with LR. This is a community probe, not a general claim.
>>>>>>> 740cb02 (Peanut v2.2 - LR-aware damping, ablation + community CIFR-10 results)

## Conclusion

Peanut v2.2 does not beat Adam by a lot. It beats Adam by 0.06% to 0.18% on most Fashion-MNIST tests. The real win is stability at high LR.

<<<<<<< HEAD
- Peanut v2 almost always has lower std than Adam and AdamW.
- AdamW collapses on some seeds (0.8803 on CNN, 0.8366 on MLP).
- Peanut v2 never collapses.

Final claim:
Peanut v2 is a more stable Adam variant with similar accuracy and similar time cost on CPU, for small CNN on Fashion-MNIST.
It is not a general SOTA optimizer.
=======
- Peanut v2 almost always has lower std than Adam and AdamW at high LR.
- AdamW collapses on some seeds (0.8803 on CNN, 0.8366 on MLP).
- Peanut v2 never collapsed in our Fashion-MNIST tests.

Final claim:
Peanut v2.2 is a more stable Adam variant with similar accuracy and similar time cost on CPU, for small CNN on Fashion-MNIST. Damping fades at low LR so it behaves close to Adam. It is not a general SOTA optimizer.
>>>>>>> 740cb02 (Peanut v2.2 - LR-aware damping, ablation + community CIFR-10 results)

## How to run

```powershell
.\venv\Scripts\Activate.ps1
python experiments\cnn_exp01.py
python experiments\cnn_exp02.py
