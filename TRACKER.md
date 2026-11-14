\# Peanut - Tracker / TODO



Last updated: 14 Nov 2026

Current phase: Phase 1 - CIFAR-10 validation

Current status: Peanut v2.2.0 shipped, README fixed, LinkedIn draft ready



\## How to use this file

\- \[ ] = TODO, \[x] = Done

\- Update Last updated date every time you finish a task

\- Do not delete done tasks, mark them \[x] so you have history

\- One phase at a time. Do not start Phase 2 until Phase 1 is \[x]



\## Achievements - Unlocked



\- \[x] Achievement 1: Peanut v1 built, beats Adam on 1 test

\- \[x] Achievement 2: Peanut v2 built, stable on 【entity-CNN¦canonical\_name=CNN】

\- \[x] Achievement 3: 11 experiments done, Fashion-MNIST, 3 seeds

\- \[x] Achievement 4: Ablation done - both changes needed, warmup collapses at high LR

\- \[x] Achievement 5: Repo public - https://github.com/dskc/peanut

\- \[x] Achievement 6: Release v2.0.0 published

\- \[x] Achievement 7: Community feedback received - Hugging Face, 2 feedbacks

\- \[x] Achievement 8: Peanut v2.2 built - LR-aware damping, mean wins at 0.003 and 0.01

\- \[x] Achievement 9: Release v2.2.0 published

\- \[x] Achievement 10: README merge conflict fixed, tail present (Results saved + Limits)

\- \[x] Achievement 11: 【entity-GitHub¦canonical\_name=GitHub】 Discussion pinned, Hugging Face topic posted

\- \[x] Achievement 12: LinkedIn post drafted

\- \[ ] Achievement 13: CIFAR-10 validated, 50 epochs, 3 seeds

\- \[ ] Achievement 14: Beats AdamW + Lion on at least 1 dataset

\- \[ ] Achievement 15: Peanut pip installable

\- \[ ] Achievement 16: Transformer test done - Tiny Shakespeare

\- \[ ] Achievement 17: Paper / blog published on Hugging Face

\- \[ ] Achievement 18: 100 stars / 10 users tried Peanut



\## Phase 0 - Done - Lock v2.2

Goal: v2.2 shipped and clean

Deadline: Done 14 Nov 2026



\- \[x] Freeze peanut.py v2.2

\- \[x] README v2.2 clean, no <<<<<<< markers

\- \[x] README tail present: Results saved + Limits

\- \[x] Push to 【entity-GitHub¦canonical\_name=GitHub】, main up to date

\- \[x] Release v2.2.0 published

\- \[x] LinkedIn draft ready



\## Phase 1 - CIFAR-10 Validation - NOW

Goal: Prove Peanut wins on 2nd dataset, not just Fashion-MNIST

Infra: Laptop for code + Colab Free for GPU

Deadline: 3 weeks - Week 1 setup, Week 2 runs, Week 3 writeup

Success: CIFAR-10 table in README, or honest failure note



\### Week 1 - Setup - TODO

\- \[ ] Create experiments/cifar10\_exp01.py on laptop

\- \[ ] TinyCNN, batch 128, 50 epochs, 3 seeds (0,1,42)

\- \[ ] Optimizers: AdamW vs Peanut v2.2, LR 0.003 and 0.01

\- \[ ] Test 1 seed on Colab, check it runs and saves CSV

\- \[ ] Save CSV to Drive after each seed



\### Week 2 - Runs - TODO

\- \[ ] Run AdamW LR 0.003, 3 seeds, save CSVs

\- \[ ] Run Peanut v2.2 LR 0.003, 3 seeds, save CSVs

\- \[ ] Run AdamW LR 0.01, 3 seeds, save CSVs

\- \[ ] Run Peanut v2.2 LR 0.01, 3 seeds, save CSVs

\- \[ ] Rule: 2 runs per Colab session max, save after each



\### Week 3 - Writeup - TODO

\- \[ ] Compute mean, std, worst for all runs on laptop

\- \[ ] Update README with CIFAR-10 table

\- \[ ] If win: Publish Release v2.3.0 - CIFAR-10 validated

\- \[ ] If lose: Publish honest note - Wins on Fashion-MNIST, loses on CIFAR-10

\- \[ ] Post result on Hugging Face topic

\- \[ ] Mark Achievement 13 \[x]



\## Phase 2 - Stronger Baselines - TODO

Goal: Beat AdamW + Lion, test transformer

Infra: Kaggle Free 30h/week GPU + Laptop for analysis

Deadline: 4 weeks after Phase 1

Success: 3 optimizers compared on 2 datasets



\### Week 4-5 - Lion + CIFAR-100

\- \[ ] Add Lion optimizer file, 30 lines

\- \[ ] CIFAR-100, TinyCNN, 50 epochs, 3 seeds

\- \[ ] Optimizers: AdamW vs Lion vs Peanut v2.2, LR 0.003

\- \[ ] Save all CSVs to Drive



\### Week 6-7 - Transformer

\- \[ ] Read AdamW paper, 1 hour

\- \[ ] Read Lion paper, 1 hour

\- \[ ] Read Sophia paper, 1 hour

\- \[ ] Tiny Shakespeare, small transformer, 20 epochs, 3 seeds

\- \[ ] Optimizers: AdamW vs Lion vs Peanut v2.2

\- \[ ] Mark Achievement 14 \[x] if Peanut beats AdamW+ Lion on 1 dataset

\- \[ ] Mark Achievement 16 \[x] when transformer done



\## Phase 3 - Make it Usable - TODO

Goal: Anyone can pip install Peanut in 1 minute

Infra: Laptop only, no GPU needed

Deadline: 2 weeks after Phase 2



\- \[ ] Create pyproject.toml, pip package peanut-optimizer

\- \[ ] Test pip install on fresh venv on laptop

\- \[ ] Add 3 examples: MLP, CNN, Transformer, each 20 lines

\- \[ ] Add benchmark script: 5 min CPU run, Adam vs Peanut

\- \[ ] Publish to PyPI

\- \[ ] Post on LinkedIn + Hugging Face: Peanut now pip installable

\- \[ ] Mark Achievement 15 \[x]



\## Phase 4 - Paper / Long Game - TODO

Goal: 4-page workshop paper, not 12-page paper

Infra: Laptop + Colab + Kaggle Free

Deadline: 2-3 months after Phase 3



\- \[ ] Hugging Face Space demo: Try Peanut vs Adam on CIFAR-10

\- \[ ] List on Papers With Code

\- \[ ] Write 4-page paper: Idea, Ablation, 3 datasets, Limits

\- \[ ] Submit to arXiv, free

\- \[ ] Write Hugging Face blog: What I learned building Peanut

\- \[ ] Mark Achievement 17 \[x]

\- \[ ] Mark Achievement 18 \[x] when 10 users tried it



\## Weekly Routine - Laptop Only

\- Monday: 1h, plan 1 experiment, write code on laptop

\- Tue-Thu: 1-2h/day, run on Colab / Kaggle, save CSV to Drive

\- Friday: 1h, analyze on laptop, update README + TRACKER.md

\- Weekend: Rest, or 1 LinkedIn / HF post



\## Rules - Do Not Break

1\. Freeze optimizer code before sweep. No tweaks mid-sweep.

2\. Same model, same data split for all optimizers.

3\. Sweep LR separately for each optimizer.

4\. 3 seeds minimum, 5 seeds for paper.

5\. If experiment > 2h on laptop, run on Colab / Kaggle, not laptop.

6\. Save CSV to Drive immediately after each seed.

7\. Publish failures too. Honest failure = trust.

8\. One phase at a time. No Phase 2 until Phase 1 is \[x].



\## Links

\- Repo: https://github.com/dskc/peanut

\- Release v2.2.0: https://github.com/dskc/peanut/releases

\- Hugging Face topic: discuss.huggingface.co - Peanut topic

\- GitHub Discussion: Pinned Welcome - Try Peanut v2

\- LinkedIn: Draft ready, not yet posted - check Phase 0

