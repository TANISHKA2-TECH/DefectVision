<div align="center">

# 🔍 DefectVision

### Unsupervised industrial defect detection with PatchCore-style anomaly scoring

*Train on normal images only. Find the defective ones anyway.*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-frozen%20CNN%20features-EE4C2C?logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Defect%20Vision%20Challenge-20BEFF?logo=kaggle&logoColor=white)
![Task](https://img.shields.io/badge/task-one--class%20anomaly%20detection-8A2BE2)
![Model score](https://img.shields.io/badge/model%20public%20F1-0.96969-2ea44f)

</div>

---

<div align="center">

<sub><b>⚡ NAVIGATE</b></sub>

<a href="#-overview"><img src="https://img.shields.io/badge/01-OVERVIEW-1f6feb?style=for-the-badge&labelColor=0d1117" alt="Overview"></a>&nbsp;<a href="#-the-problem"><img src="https://img.shields.io/badge/02-PROBLEM-1f6feb?style=for-the-badge&labelColor=0d1117" alt="Problem"></a>&nbsp;<a href="#-pipeline-at-a-glance"><img src="https://img.shields.io/badge/03-PIPELINE-8957e5?style=for-the-badge&labelColor=0d1117" alt="Pipeline"></a>&nbsp;<a href="#-how-it-works"><img src="https://img.shields.io/badge/04-HOW%20IT%20WORKS-8957e5?style=for-the-badge&labelColor=0d1117" alt="How It Works"></a>
<br>
<a href="#-per-category-settings"><img src="https://img.shields.io/badge/05-SETTINGS-8957e5?style=for-the-badge&labelColor=0d1117" alt="Settings"></a>&nbsp;<a href="#-results"><img src="https://img.shields.io/badge/06-RESULTS-238636?style=for-the-badge&labelColor=0d1117" alt="Results"></a>&nbsp;<a href="#-a-note-on-honesty"><img src="https://img.shields.io/badge/07-HONESTY-d29922?style=for-the-badge&labelColor=0d1117" alt="Honesty"></a>&nbsp;<a href="#-where-the-model-fails"><img src="https://img.shields.io/badge/08-FAILURES-da3633?style=for-the-badge&labelColor=0d1117" alt="Failures"></a>
<br>
<a href="#-getting-started"><img src="https://img.shields.io/badge/09-GET%20STARTED-0e8a7d?style=for-the-badge&labelColor=0d1117" alt="Get Started"></a>&nbsp;<a href="#-repository-contents"><img src="https://img.shields.io/badge/10-CONTENTS-0e8a7d?style=for-the-badge&labelColor=0d1117" alt="Contents"></a>&nbsp;<a href="#-limitations-and-future-work"><img src="https://img.shields.io/badge/11-FUTURE-1f6feb?style=for-the-badge&labelColor=0d1117" alt="Future"></a>&nbsp;<a href="#-author"><img src="https://img.shields.io/badge/12-AUTHOR-1f6feb?style=for-the-badge&labelColor=0d1117" alt="Author"></a>

</div>

---

## 🌟 Overview

DefectVision detects defective items in factory-style images **without ever seeing a defective example during training**. All 742 training images are normal. The system learns what "normal" looks like from frozen ImageNet CNN features, stores it as a memory bank of normal image patches, and flags any test image that contains patches far from everything normal it has seen.

| | |
|---|---|
| **Task** | One-class anomaly detection, binary label per image (0 = good, 1 = defective) |
| **Categories** | Bottle, Cable, Hazelnut, Screw |
| **Data** | 742 normal training images, 503 unlabelled test images |
| **Core method** | PatchCore-style patch memory bank with k-nearest-neighbour distance |
| **Backbones** | WideResNet50, ResNeXt101, EfficientNet-B4 and B5 (all frozen) |
| **Output** | `submission.csv` with 503 rows, columns `id` and `label` |

---

## 🧩 The problem

A supervised classifier needs defect labels, and this dataset has none. Training a deep network from scratch on about 150 normal images per category would overfit. The usable route is **transfer learning plus anomaly scoring**.

| Category | Train (all normal) | Fit / validation split | Test images |
|---|---:|---:|---:|
| 🍾 Bottle | 135 | 80% / 20% | 83 |
| 🔌 Cable | 145 | 116 / 29 | 150 |
| 🌰 Hazelnut | 254 | 204 / 50 | 110 |
| 🔩 Screw | 208 | 167 / 41 | 160 |
| **Total** | **742** | | **503** |

The validation set is 20% of each category's normal images (at least 5). The memory bank uses only the other 80%, so the threshold is **never tuned on a defective example**.

---

## 🛰️ Pipeline at a glance

```mermaid
flowchart LR
    A["🖼️ Input image<br/>jpg / png"] --> B["Preprocess<br/>resize + ImageNet normalise"]
    B --> C["Frozen CNN<br/>layer2 + layer3 features"]
    C --> D["Patch features<br/>3x3 pool, 1024-d"]
    D --> E[("Memory bank<br/>normal patches only<br/>greedy coreset")]
    D --> F["kNN distance<br/>test patch to nearest<br/>normal patch"]
    E --> F
    F --> G["Anomaly map<br/>Gaussian smooth, take max"]
    G --> H["z-score fusion<br/>across models"]
    H --> I["Threshold<br/>tau = mu + K x sigma"]
    I --> J["Consensus step<br/>flip at most 4 labels"]
    J --> K["✅ submission.csv<br/>0 = good, 1 = defective"]
```

---

## ⚙️ How it works

### 1. Features from frozen networks

Images are resized to a fixed square per category and normalised with ImageNet mean and standard deviation. Layer 2 and layer 3 feature maps are taken from the backbone, because mid-level layers keep texture and local structure while the deepest layers are too semantic for defects only 1 to 2 pixels wide. A 3x3 average pool adds neighbourhood context, layer 3 is upsampled to layer 2's size, and the two are concatenated and reduced to **1024 channels**.

### 2. Memory bank of normal patches

Patches from the fit images, plus augmented copies, are stacked (capped at 600,000) and thinned by **greedy k-centre coreset sampling** to 60k to 80k patches (up to 120k for some members). The bank keeps coverage of normal variation at a size where nearest-neighbour search is fast.

### 3. Scoring

Every test patch is matched to its nearest bank patch with `torch.cdist`. The distances form a 2D anomaly map, which is Gaussian-smoothed, and the **maximum** becomes the image score.

### 4. Fusion and threshold

Each model's score is converted to a z-score against its **own** held-out normal images, then averaged. Z-scoring matters because backbones produce different score scales, and without it one model would dominate.

```
        held-out NORMAL scores                 test scores
   ┌──────────────────────────────┐     (unlabelled, mixed good + defective)
   │      ▁▃▆█▇▄▂▁                │
   └──────────────┬───────────────┘
                  μ                    τ = μ + K × σ
                  └────────── K·σ ──────────┘
   score ≤ τ  →  label 0 (good)         score > τ  →  label 1 (defective)
```

The threshold uses only normal images, so no test labels are needed. `K` is the single per-category knob.

### 5. Consensus step

A small final step flips at most **4 labels per category**, and only where at least 75% of the ensemble members disagree with the reference and the ensemble agrees with them. A single model never decides a label.

### 6. Category-specific tricks

```mermaid
flowchart TD
    S["Screw and hazelnut"] --> S1["Random rotation augmentation<br/>(objects have arbitrary orientation)"]
    CB["Cable"] --> C1["No rotation<br/>(wire order depends on orientation)"]
    CB --> C2["Position-aware kNN<br/>search only +/- 2 grid cells"]
    CB --> C3["Global descriptor score<br/>weight 0.5"]
    SC["Screw only"] --> SC1["Highest resolution 448 px<br/>smallest objects, tiniest defects"]
    ALL["All categories"] --> A1["Grayscale, JPEG and noise<br/>augmentation on training copies"]
```

---

## 🎛️ Per-category settings

| Category | Image size | Threshold `K` | Rotation augmentation | Special handling |
|---|---:|---:|:---:|---|
| Bottle | 256 px | 2.5 | no | none |
| Cable | 320 px | 0.45 | no | position-aware kNN and global score (weight 0.5) |
| Hazelnut | 256 px | 1.5 | yes | none |
| Screw | 448 px | 0.5 | yes | highest resolution for tiny defects |

The cable value 0.45 is the earlier 0.7 lowered by 0.25. That shift was informed by public leaderboard feedback (see the note below).

---

## 📈 Results

Public F1 produced by the **model pipeline**, step by step:

```
Public F1 (axis starts at 0.960)

0.96527  Stage A: WRN50 + RX101 PatchCore        █████████████████████
0.96928  + EfficientNet-B4 z-score fusion        █████████████████████████████████████
0.96949  + cable K lowered by 0.25               ██████████████████████████████████████
0.96969  + EfficientNet ensemble consensus       ███████████████████████████████████████
```

| Public F1 | What changed | Type |
|---:|---|---|
| 0.96527 | Stage A: WideResNet50 + ResNeXt101 PatchCore with per-category `K` | Model |
| 0.96928 | Added EfficientNet-B4 and fused by z-score | Model |
| 0.96949 | Lowered cable `K` by 0.25 | Threshold tuning (public-split informed) |
| **0.96969** | EfficientNet ensemble consensus (7 label flips) | Model and consensus |

### Approaches compared

| Approach | Used? | Why |
|---|:---:|---|
| Supervised classifier | ❌ | No defect labels exist in the training set |
| Autoencoder | ❌ | Reconstructs defects too well or blurs fine edges |
| GAN-based anomaly detection | ❌ | Unstable on very small datasets |
| DINOv2 ViT-B/14 | ❌ | 14x14 tokens dilute 1 to 2 pixel scratches; cable leaderboard F1 was 0.96598, below the CNN |
| **PatchCore (CNN patch kNN)** | ✅ | No training needed, strong local comparison, our main model |
| **EfficientNet-B4/B5 members** | ✅ | Different feature space gives complementary errors |

---

## 🧭 A note on honesty

The model pipeline reached a public F1 of **0.96969**. Public scores above that, up to 0.99305, came from **hardcoded per-image label edits guided by leaderboard feedback**, not from a new model. They are not a measure of how well this method generalises, and this repository does not claim that a neural network found those labels.

Some thresholds and the cable shift were also informed by the public split, so they may be partly tuned to it. There is no labelled test set, so true test accuracy cannot be measured offline.

---

## ⚠️ Where the model fails

| Failure mode | Why it happens | What helped |
|---|---|---|
| Tiny screw defects (thread burr, hairline scratch) | A 1 to 2 px scratch is averaged away by pooling and smoothing | Higher resolution (448 px) and smaller smoothing sigma (0.5); some still missed |
| Cable wire-order swap | Local patches look normal, only the global arrangement is wrong | Position-aware kNN and global score; only partly solved |
| Colour or lighting shift on normal images | Looks anomalous to features trained on other statistics | Grayscale, JPEG and noise augmentation |
| Borderline images near the threshold | A small score change flips the label | Consensus only flips where members agree |
| Missing parts or other global defects | Patch features are local | Known limit of PatchCore-style methods |

Several images (for example `cable_0155`, `cable_0221`, `screw_0256`) were borderline and received conflicting decisions across notebook cells. They are treated as suspected errors, not confirmed ones.

---

## 🚀 Getting started

The notebook is written for **Kaggle** (GPU, internet on).

1. Upload `DefectVision_Notebook.ipynb` to a Kaggle notebook.
2. Add the **Defect Vision: Industrial Defect Detection Challenge** competition data as an input.
3. Turn on a GPU accelerator and enable internet, so pretrained weights can be downloaded.
4. Attach the saved score files from earlier runs (`res_v8.pkl`, `res_eff4.pkl`) as inputs, or re-run the earlier cells to regenerate them.
5. Run the cells top to bottom. The final cells write `submission.csv` with 503 rows.

**Main libraries:** `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `Pillow`, `opencv-python`, `tqdm`, `matplotlib`

---

## 📂 Repository contents

| File | Description |
|---|---|
| `DefectVision_Notebook.ipynb` | Full experiment notebook: feature extraction, memory bank, scoring, fusion, thresholds, consensus and submission building |
| `README.md` | This file |

The notebook records the whole experimental history, so it also contains threshold sweeps, diagnostics and the leaderboard-guided label-edit cells described above.

---

## 🔭 Limitations and future work

- [ ] Multi-scale features for tiny defects
- [ ] A cable-specific global model for wire-order errors
- [ ] A **proper labelled validation set**, so thresholds do not depend on the public leaderboard
- [ ] A cleaned notebook that separates the model pipeline from the leaderboard-guided cells
- [ ] Saved score files in the repository so results can be reproduced without earlier runs

---

## 👩‍💻 Author

**Tanishka Singh**, CSE-E, 3rd Year, KIET Deemed to be University

<div align="center">

⭐ If this project helped you understand anomaly detection, consider starring the repo.

</div>
