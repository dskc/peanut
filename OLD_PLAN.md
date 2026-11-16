# Peanut - OLD PLAN - Till 15 Nov 2026 - DO NOT EXECUTE - For reference only
Saved on 16 Nov 2026. Replaced by TRACKER.md revised plan.

## What it was
Prove Peanut v2.2 is good by testing it bigger.

## Old goals
- Prove v2.2 wins on CIFAR-10
- Beat AdamW + Lion
- Make pip package
- Write paper

## Old experiments
- CIFAR-10, TinyCNN, 50 epochs, batch 128, AdamW vs Peanut v2.2, LR 0.003 and 0.01
- 3 seeds first (0,1,42), then 5 seeds (0,1,42,123,999)
- Then 100-epoch run to check if lead persists
- Then CIFAR-100, TinyCNN, 50 epochs
- Then Tiny Shakespeare transformer, 20 epochs

## Old baselines
Adam, AdamW, SGD+Momentum, RMSprop, Peanut v2, DeepSpeed CPUAdam, Lion, Sophia

## Old datasets
MNIST, Fashion-MNIST, CIFAR-10 ResNet-18, California Housing, 【entity-IMDB¦canonical_name=IMDB】

## Old features - V4 file as written
Adaptive momentum, adaptive smoothing, adaptive LR, adaptive clipping, decoupled WD, FP16 states, factored states, curvature approx, vectorized ops, lower precision - all listed together

## Old benchmark size
ResNet-18, CIFAR-10, 10 seeds, 6 optimizers, 100 epochs, 60 total runs

## Old framework
Separate files per experiment: cnn_exp01.py, cnn_exp02.py, mlp_m_test.py, cifar10_exp01.py - different code per optimizer

## Reviewer checklist - was next in old plan
1. CIFAR-10 vs Adam/AdamW
2. Training time on identical CPU
3. 5-10 seeds
4. 100+ epochs persistence

## Why it was replaced
Too big, too many datasets, too many features at once, too many runs (60), ResNet-18 too heavy for laptop, different code per experiment caused m-test bug, would take months and risk quitting before finishing.
Replaced by revised plan in TRACKER.md: Start small, one pipeline, one feature at a time, 9 runs first.