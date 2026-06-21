# Traffic Sign Recognition Using CNN
### SRM Institute of Science and Technology | CINTEL Department | Minor Project 21CSP302L

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.68%25-green)
![Parameters](https://img.shields.io/badge/Parameters-2.1M-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB-yellow)

---

## Overview

This project trains a compact five-block Convolutional Neural Network (CNN)
on the German Traffic Sign Recognition Benchmark (GTSRB) dataset to
automatically classify road traffic signs into 43 categories.

The model is designed for real-world ADAS deployment — not just benchmark
accuracy. It uses CLAHE preprocessing to handle lighting variation, online
data augmentation to generalise across camera angles and distances, and
Grad-CAM visualisations to validate that the model attends to the correct
sign regions.

**Authors:** Shaik Nafeez, Shaik Sayyed Baji
**Guide:** Dr. Lakshmi Narayanan K, Assistant Professor, CINTEL
**Institution:** SRM Institute of Science and Technology, Chennai

---

## Results

| Condition | Accuracy |
|---|---|
| Clean Test Set (12,630 images) | **98.68%** |
| Gaussian Noise (σ = 0.08) | 95.8% |
| Low Brightness (50% reduction) | 98.0% |
| Centre-Patch Occlusion (8×8 px) | 31.8% |

| Metric | Value |
|---|---|
| Test Loss | 0.0579 |
| Macro F1-Score | 0.9831 |
| Total Parameters | ~2.1 Million |
| Training Epochs | 30 |

---

## Model Architecture
