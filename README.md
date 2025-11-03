# Semi-Supervised Image Segmentation with ResNet-18 and Pseudo-Labeling

This project builds a **semi-supervised image segmentation pipeline** that classifies and segments satellite imagery using a limited amount of labeled data and a large set of unlabeled samples.

The model integrates **ResNet-18** as a feature encoder, **Dice + Binary Cross-Entropy loss**, and **pseudo-labeling** to leverage unlabeled data effectively. The approach achieved a **mean IoU of ≈ 0.46** on a held-out test set.

---

## 🌎 Project Overview
- **Goal:** Predict binary segmentation masks for satellite images.  
- **Challenge:** Only a small portion of the dataset is labeled; most images are unlabeled.  
- **Solution:** A semi-supervised training pipeline that iteratively generates and refines pseudo-labels to improve performance.  
- **Metric:** Mean Intersection-over-Union (mIoU).  

---

## 🧠 Methodology

### 1️⃣ Architecture
- **Backbone:** ResNet-18 pretrained on ImageNet  
- **Decoder:** Lightweight up-sampling head with 1×1 convolution + sigmoid  
- **Loss:** Combined **Dice Loss + Binary Cross-Entropy (BCE)**  
- **Optimizer:** AdamW; **Scheduler:** ReduceLROnPlateau

### 2️⃣ Semi-Supervised Training Pipeline
1. **Supervised Warm-Up** – Train on the small labeled set.  
2. **Pseudo-Label Generation** – Predict masks on unlabeled data; keep high-confidence predictions.  
3. **Joint Training** – Train on (labeled + pseudo-labeled) together.  
4. **Fine-Tuning** – Retrain on labeled data only to denoise.

### 3️⃣ Data Augmentation
Random flips/rotations, color jitter, normalization (ImageNet mean/std), random crops/resizing.

### 4️⃣ Inference
**Test-Time Augmentation (TTA)** with flips/rotations and prediction averaging. Threshold = 0.5.

---

## 📊 Results

| Phase | Description | mIoU (Validation) |
|:------|:------------|:-----------------:|
| Supervised Baseline | Labeled data only | 0.37 |
| + Pseudo-Labeling   | Add confident unlabeled | 0.43 |
| Final (+ TTA)       | Fine-tuned & averaged | **0.46** |

> Pseudo-labeling improved validation mIoU by roughly **+0.09** over the baseline.

---

## 📂 Repository Structure
<pre><code>.
├── notebooks/
│   └── Semi-Supervised_Project.ipynb      # full training + inference notebook
├── submission/
│   └── submission.csv                  # final predictions (competition format)
├── Projec_Report.pdf           # detailed methodology & results
└── README.md
</code></pre>

---

## ▶️ How to Run

### Option 1 — Google Colab
Upload the notebook and run cell-by-cell.  
Ensure the dataset directories are accessible in your environment (Drive/Kaggle):  
`train-semi/`, `train-semi-segmentation/`, `unlabeled/`, `test/`.

### Option 2 — Local (with GPU)
```bash
jupyter notebook notebooks/CSE164_Final_Project.ipynb
