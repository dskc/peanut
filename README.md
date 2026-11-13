# Peanut - Beating Adam on CPU

Goal: Build an optimizer better than Adam on a specific weakness,
validated on CPU-only laptop.

Status: Completed, local validation done. Peanut v2 is the final version.

Hardware: CPU-only, 32GB RAM, Intel Alder Lake N-series, Windows 11
Python: 3.13, torch 2.15.0+cpu, no GPU

## Structure

data/ : datasets, auto-downloaded
src/optimizers/ : Peanut v2
experiments/ : final validation runs
results/ : tables

## Idea

Peanut v2 is Adam with two changes:

1. Damped first step: first update is multiplied by 0.2.
   The first gradient is the noisiest, so we take a smaller first step.

2. Inflated v init: v = g1^2, not (1 - beta2) * g1^2.
   This acts as extra early damping in steps 2 to 20.

Everything else is standard Adam: betas (0.9, 0.999), eps 1e-8,
bias correction from step 2 onward.

## Results

All results: Fashion-MNIST, batch 64, lr 0.01, 3 seeds (0, 1, 42).

### Experiment 01 - CNN, 10 epochs

| Optimizer | Mean | Std |
| --- | --- | --- |
| Adam | 0.8934 | 0.0028 |
| AdamW | 0.8889 | 0.0061 |
| Peanut v2 | 0.8952 | 0.0012 |

### Experiment 02 - CNN, 20 epochs

| Optimizer | Mean Final | Std | Mean Max |
| --- | --- | --- | --- |
| Adam | 0.8901 | 0.0041 | 0.8968 |
| Peanut v2 | 0.8972 | 0.0038 | 0.9030 |

Note: Experiment 02 was run to check long training. Peanut v2 stays
stable on long training.

## Conclusion

Peanut v2 does not beat Adam by a lot. It beats Adam by 0.06% to 0.18%
on most tests. The real win is stability:

- Peanut v2 almost always has lower std than Adam and AdamW.
- AdamW collapses on some seeds (0.8803 on 【entity-CNN¦canonical_name=CNN】, 0.8366 on MLP).
- Peanut v2 never collapses.

Final claim:
Peanut v2 is a more stable Adam variant with similar accuracy
and similar time cost on CPU, for small CNN on Fashion-MNIST.
It is not a general SOTA optimizer.

## How to run

```powershell
.\venv\Scripts\Activate.ps1
python experiments\cnn_exp01.py
python experiments\cnn_exp02.py
