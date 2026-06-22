# 🧠 Brain Tumor MRI Classification

A deep learning project that classifies brain MRI scans into 4 tumor categories using PyTorch. Compares a custom CNN against a Feedforward Neural Network (FFNN) baseline, with an interactive Streamlit web app for real-time inference.

---

## 📌 Problem Statement

Brain tumor diagnosis from MRI scans is time-consuming and requires specialized expertise. This project automates tumor classification using deep learning, helping assist radiologists with faster and more consistent diagnoses.

---

## 🗂️ Dataset

- **Source:** [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** `glioma` | `meningioma` | `notumor` | `pituitary`
- **Split:** Training / Validation (80:20) / Testing

---

## 🏗️ Model Architectures

### CNN (Primary Model)
```
Input (3×128×128)
→ Conv2d(3→32) + BatchNorm + ReLU + MaxPool
→ Conv2d(32→64) + BatchNorm + ReLU + MaxPool
→ Conv2d(64→128) + BatchNorm + ReLU + MaxPool
→ Conv2d(128→256) + BatchNorm + ReLU + MaxPool
→ Flatten → Linear(256×8×8 → 512) + BatchNorm + ReLU
→ Linear(512 → 256) + ReLU + Dropout(0.3)
→ Linear(256 → 4)
```

### FFNN (Baseline)
```
Input (128×128×3 flattened)
→ Linear(49152 → 1024) + BatchNorm1d + ReLU + Dropout(0.4)
→ Linear(1024 → 512) + BatchNorm1d + ReLU + Dropout(0.4)
→ Linear(512 → 4)
```

---

## ⚙️ Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping | patience=10 |
| Batch Size | 32 |
| Max Epochs | 30 |
| Input Size | 128×128 |

**Augmentation (training only):**
- RandomHorizontalFlip
- RandomRotation(15°)
- RandomResizedCrop(scale=0.8–1.0)
- Normalize (mean=0.5, std=0.5)

---

## 📊 Results

| Model | Accuracy | Training Time | Inference Time |
|-------|----------|--------------|----------------|
| **CNN** | **~93%** | 1621.96 seconds | 7.3202 seconds |
| FFNN | ~76% | 1273.77 seconds |  5.4022 seconds |

"CNN significantly outperforms FFNN in accuracy, although it requires more training and inference time due to the computational complexity of convolutional layers."
---

## 🖥️ Streamlit App

An interactive web app that lets you upload an MRI image and get a real-time classification prediction.

```bash
streamlit run app.py
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/omaralimona05-maker/brain-tumor-cnn.git
cd brain-tumor-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place it as:
```
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Train the models
```bash
python Neural_project.ipynb  # or run in Jupyter
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellowgreen)

---

## 📁 Project Structure

```
brain-tumor-cnn/
├── Neural_project.ipynb   # Training & evaluation notebook
├── app.py                 # Streamlit web app
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Ali Yasser** — Intelligent Systems Student, Alexandria National University  
GitHub: [@omaralimona05-maker](https://github.com/omaralimona05-maker)
