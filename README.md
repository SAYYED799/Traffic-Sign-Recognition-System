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

---

## Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**
- 51,839 colour images across 43 sign categories
- 39,209 training images / 12,630 test images
- Download: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

---

## Project Structure

---

## Setup and Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-recognition-cnn.git
cd traffic-sign-recognition-cnn
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the dataset

**Option A — Kaggle API (recommended)**
```bash
# Place your kaggle.json in ~/.kaggle/ first
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign \
  -p gtsrb_data --unzip
```

**Option B — Manual download**
Download from:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Extract so the structure looks like:

### Step 4 — Train the model
```bash
python tsr_cnn_model.py
```

Or open the notebook:
```bash
jupyter notebook tsr_cnn_notebook.ipynb
```

---

## Running on Google Colab (Recommended)

1. Open https://colab.research.google.com
2. Upload `tsr_cnn_notebook.ipynb`
3. Set runtime to **T4 GPU** (Runtime → Change runtime type)
4. Run all cells top to bottom

Training takes approximately **5–8 minutes** with T4 GPU.

---

## Test Your Own Image

```python
from tensorflow import keras
import cv2
import numpy as np

# Load saved model
model = keras.models.load_model("tsr_cnn_model.h5")

# Class names
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50",
    "Speed limit 60", "Speed limit 70", "Speed limit 80",
    "End speed 80", "Speed limit 100", "Speed limit 120",
    "No passing", "No passing >3.5t", "Right-of-way",
    "Priority road", "Yield", "Stop",
    "No vehicles", "No trucks", "No entry",
    "General caution", "Curve left", "Curve right",
    "Double curve", "Bumpy road", "Slippery road",
    "Narrows right", "Works ahead", "Traffic signals",
    "Pedestrians", "Children", "Bicycles",
    "Ice/snow", "Wild animals", "End restrictions",
    "Turn right", "Turn left", "Go straight",
    "Straight/right", "Straight/left", "Keep right",
    "Keep left", "Roundabout", "End no passing",
    "End no passing >3.5t"
]

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32)).astype(np.float32) / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred  = np.argmax(probs)
    print(f"Predicted : {CLASS_NAMES[pred]}")
    print(f"Confidence: {probs[pred]*100:.2f}%")

predict("your_sign_photo.jpg")
```

---

## Key Findings

**What worked well:**
- CLAHE preprocessing added +1.4% on low-brightness images at zero inference cost
- Data augmentation reduced train-val gap from 3.2% to 0.8%
- 98.0% accuracy retained under 50% brightness reduction
- 95.8% accuracy retained under Gaussian noise
- Grad-CAM confirmed correct attention on sign-specific regions

**What needs future work:**
- Centre-patch occlusion dropped accuracy to 31.8% — the CNN
  concentrates on central sign features and fails when they are blocked
- Fix: occlusion-aware augmentation + spatial attention module (CBAM)
- Model trained on German signs only — Indian road fine-tuning needed

---

## Comparison Against Baselines

| Method | Accuracy | Parameters | Jetson Nano Ready |
|---|---|---|---|
| HOG + SVM | 92.1% | — | Yes |
| Decision Tree + HOG | 90.4% | — | Yes |
| Multi-Scale CNN (Sermanet) | 98.31% | 2.5M | Possible |
| VGG-16 Transfer Learning | 98.9% | 138M | No |
| Multi-Column DNN (Ciresan) | 99.46% | 50M | No |
| **Proposed CNN (Ours)** | **98.68%** | **2.1M** | **Yes** |

---

## Future Work

- [ ] Occlusion-aware augmentation (random patches during training)
- [ ] CBAM spatial attention module
- [ ] YOLO v8 integration for detection + classification pipeline
- [ ] TensorRT conversion for Jetson Nano deployment
- [ ] Indian road sign dataset collection and fine-tuning
- [ ] Higher input resolution (64×64) training

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Citation

If you use this project in your work, please cite:
