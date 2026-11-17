# Peanut - Progress Report / Release Notes

This is the dated log of what we did, what we tested, and what we learned.
ROADMAP.md is where we are going. This file is what we have done.
README.md has the summary. This file has the story.

## How to read this

Each entry has:
- Date
- What we did
- Setup: dataset, model, seeds, epochs
- Result: numbers
- Decision: what we kept, what we dropped, what we learned

We record failures too. We do not hide them.

---

## 2026-11-17 - ROADMAP.md + PROGRESS.md published

**What:** Published direction and progress log.
**Why:** To tell community our direction and ask for comments, before adding new features.
**Decision:** CPU-only is a design decision, not a shortcoming. Next target is to close 4 unproven gaps: consistency across datasets, win over properly tuned AdamW, robustness across architectures, statistical significance.
**Links:** ROADMAP.md, PROGRESS.md

---

## 2026-11-16 - Fashion-MNIST baseline - Peanut v2.2.3 local

**What:** Ran same baseline as MNIST, but on Fashion-MNIST, to prove Peanut on 2nd dataset.
**Setup:** MLP, 3 seeds (0, 1, 42), 10 epochs, batch 64, CPU only, Peanut v2.2.3 local, Adam, AdamW, Peanut.
**Result:**
- Adam: mean 0.8605, best 0.8665, worst 0.8556, time 230.0s, RAM 383.3 MB
- AdamW: mean 0.8552, best 0.8577, worst 0.8537, time 244.1s, RAM 383.4 MB
- Peanut: mean 0.8611, best 0.8651, worst 0.8553, time 216.9s, RAM 383.7 MB
**Decision:** Peanut wins on mean (+0.06% over Adam, +0.59% over AdamW) and is fastest. Kept as evidence. Gap remains: AdamW was not separately tuned.

---

## 2026-11-15 - MNIST baseline - Peanut v2.2.3 local

**What:** Ran first clean baseline on MNIST with frozen Peanut v2.2.
**Setup:** MLP, 3 seeds (0, 1, 42), 10 epochs, batch 64, CPU only, Adam, AdamW, Peanut.
**Result:**
- Adam: mean 0.9787, best 0.9793, worst 0.9780, time 218.8s, RAM 383.2 MB
- AdamW: mean 0.9793, best 0.9796, worst 0.9788, time 227.4s, RAM 383.4 MB
- Peanut: mean 0.9806, best 0.9810, worst 0.9802, time 203.9s, RAM 383.7 MB
**Decision:** Peanut wins on mean (+0.19% over Adam, +0.13% over AdamW), with smallest std (0.0004) and best worst-seed. Kept as primary evidence.

---

## 2026-11-14 - Zero-m vs Adam-m test - Peanut v2.2

**What:** Tested if first-step m should start at zero or at Adam's first gradient.
**Setup:** Fashion-MNIST, MLP, 3 seeds, 10 epochs, Peanut with zero-m vs Peanut with Adam-m.
**Result:**
- Peanut shipped (zero-m): mean 0.8579, best 0.8591, worst 0.8539
- Peanut with Adam-m: mean 0.8577, best 0.8590, worst 0.8553
- Difference: 0.0002 mean, not significant. Adam-m helps worst seed, hurts mean slightly.
**Decision:** Zero-m carries nothing for mean. Adam-m is candidate for v2.3, but NOT merged yet. Code change paused until baseline is frozen. Recorded as open question.

---

## 2026-11-12 - Ablation experiment - Peanut v2.1

**What:** Tested which part of Peanut actually helps: damp alone, inflated v alone, or both.
**Setup:** Fashion-MNIST, 3 seeds, 10 epochs, Adam vs Peanut variants, LR 0.01 and 0.05.
**Result:**
- At LR 0.01: All optimizers ~0.86, no difference.
- At LR 0.05: Adam collapses, Peanut with both changes holds accuracy, damp alone does not, inflated v alone does not.
**Decision:** Both changes together are needed. Neither alone beats Adam. This is why Peanut works at high LR. File: results/mlp_ablation.csv

---

## 2026-11-10 - Peanut v2.2 released on 【entity-GitHub¦canonical_name=GitHub】

**What:** Published Peanut v2.2.0 on 【entity-GitHub¦canonical_name=GitHub】, made repo public.
**Setup:** LR-aware Peanut, kept as best version. v2.3 threshold version tested and discarded as buggy, reverted to v2.2.
**Decision:** v2.2 is best. v2.3 discarded. v2.2.3 is local with patches, not yet released.

---

## 2026-11-08 - Peanut v2.0.0 released

**What:** First public release, with Hugging Face topic and 【entity-GitHub¦canonical_name=GitHub】 Discussion.
**Decision:** Repo made public, About filled, Discussion pinned "Welcome - Try Peanut v2".

---

## Template for next entries - Copy this for each new experiment

### YYYY-MM-DD - [Title] - Peanut vX.X

**What:**
**Setup:** Dataset, Model, Seeds, Epochs, Batch, CPU only, Optimizers
**Result:** Mean, Std, Best, Worst, Time, RAM
**Decision:** Kept / Dropped / Learned
**File:** experiments/xxx.py, results/xxx.csv