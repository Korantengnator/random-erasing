# Random Erasing Data Augmentation — Replication Study

Replication of **"Random Erasing Data Augmentation"** (Zhong et al., AAAI 2020).  
Course: ICS 555 — Graduate Computer Vision, Ashesi University.

---

## Project Structure

```
random-erasing/
├── data/                          ← CIFAR-10 downloads here automatically
├── models/
│   └── resnet.py                  ← ResNet-18 adapted for CIFAR-10
├── transforms/
│   └── random_erasing.py          ← Core RE implementation (paper Algorithm 1)
├── results/                       ← All saved models, histories, and plots
├── train.py                       ← Experiment 0: baseline vs RE
├── evaluate.py                    ← Plot training curves for Experiment 0
├── experiment_fill_values.py      ← Experiment 1: fill value ablation
├── experiment_augmentations.py    ← Experiment 2: augmentation combinations
├── experiment_occlusion.py        ← Experiment 3: occlusion robustness
├── test_setup.py                  ← Quick sanity check before training
└── requirements.txt               ← Python dependencies
```

---

## Setup

**Step 1 — Create a virtual environment and install dependencies:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio numpy matplotlib tqdm
```

**Step 2 — Verify setup (takes ~30 seconds, no training):**

```bash
python test_setup.py
```

You should see: `All tests passed! You are ready to train.`

---

## Running Each Experiment

> **Important:** Always activate the virtual environment first:
>
> ```bash
> source venv/bin/activate
> ```
>
> Keep your laptop plugged in and prevent sleep:

---

### Experiment 0 — Baseline vs Random Erasing (Main Result)

**Replicates:** Table 1 in the paper (CIFAR-10 classification results)  
**What it does:** Trains two models for 100 epochs — one with standard augmentation (baseline) and one with Random Erasing added — and compares their test error rates.  
**Runtime:** ~40 minutes on Apple M4

```bash
python train.py
```

**Output:**

- `results/baseline_history.json` — training history for baseline
- `results/with_RE_history.json` — training history for RE model
- `results/baseline_model.pth` — saved baseline model weights
- `results/with_RE_model.pth` — saved RE model weights
- Final comparison printed to terminal

**To generate plots after training:**

```bash
python evaluate.py
```

Saves `results/training_curves.png`.

---

### Experiment 1 — Fill Value Ablation

**Replicates:** Table 3 in the paper  
**What it does:** Tests four ways to fill the erased rectangle — random values (RE-R), ImageNet mean (RE-M), zeros (RE-0), and 255s (RE-255) — to see which works best.  
**Runtime:** ~1.5 hours on Apple M4 (5 runs × ~18 min)

```bash
python experiment_fill_values.py
```

**Output:**

- `results/fill_baseline_history.json`
- `results/fill_random_history.json`
- `results/fill_mean_history.json`
- `results/fill_zero_history.json`
- `results/fill_max_history.json`
- `results/fill_comparison.png` — bar chart comparing all fill modes
- Summary table printed to terminal

---

### Experiment 2 — Augmentation Combinations

**Replicates:** Table 5 in the paper  
**What it does:** Tests all 8 combinations of Random Flipping (RF), Random Cropping (RC), and Random Erasing (RE) to show the methods are complementary.  
**Runtime:** ~2.5 hours on Apple M4 (8 runs × ~18 min)

```bash
python experiment_augmentations.py
```

**Output:**

- `results/aug_None_history.json`
- `results/aug_RF_history.json`
- `results/aug_RC_history.json`
- `results/aug_RE_history.json`
- `results/aug_RF+RC_history.json` _(stored as aug_RFandRC_history.json)_
- `results/aug_RF+RE_history.json`
- `results/aug_RC+RE_history.json`
- `results/aug_RF+RC+RE_history.json`
- `results/augmentation_comparison.png` — bar chart
- Summary table printed to terminal

**To regenerate plots or recover results after closing the terminal:**

```bash
python recover_aug_results.py
```

---

### Experiment 3 — Occlusion Robustness

**Replicates:** Figure 3 in the paper  
**What it does:** Loads the already-trained baseline and RE models from Experiment 0 and evaluates them on test images with a black centre patch of increasing size (0% to 50% of the image). No new training required.  
**Prerequisite:** You must have run `train.py` first (Experiment 0).  
**Runtime:** ~15 minutes on Apple M4

```bash
python experiment_occlusion.py
```

**Output:**

- `results/occlusion_robustness.png` — line plot comparing both models across occlusion levels
- Full results table printed to terminal

---

### Run All Experiments Back-to-Back

To queue all experiments and run them overnight:

```bash
python experiment_fill_values.py && python experiment_augmentations.py
```

Then after those finish:

```bash
python experiment_occlusion.py
```

---

## Expected Results Summary

| Experiment                | Key Finding                                          |
| ------------------------- | ---------------------------------------------------- |
| Baseline vs RE            | RE reduces error: 9.21% → 8.81%                      |
| Fill values               | RE-M (mean) best at 7.97%; RE-0 barely helps (9.25%) |
| Augmentation combinations | RF+RC+RE best at 8.81%; all methods complementary    |
| Occlusion robustness      | At 50% occlusion: baseline 84.46% vs RE 32.51%       |

---

## Reproducibility Notes

- All random seeds are fixed at `42`. Running any experiment twice produces identical results.
- We trained for **100 epochs** instead of the paper's 300. Learning rate milestones (epochs 150, 225) are never reached, so all runs use a constant lr=0.1. This explains why our absolute error rates are higher than the paper's reported values.
- We use **standard ResNet-18** adapted for CIFAR-10, not the pre-activation ResNet-18 used in the paper.
- All directional findings match the paper despite these deviations.

---
