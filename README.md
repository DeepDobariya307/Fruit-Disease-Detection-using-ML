# 🍎 Fruit Disease Detection using Deep Learning

> **VGG-16 CNN-based image classification model for automated early detection of fruit diseases — published at IEEE ICCCIS 2023.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![IEEE](https://img.shields.io/badge/Published-IEEE%20ICCCIS%202023-blue?logo=ieee)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-96--98%25-brightgreen)

---

## 📌 Project Overview

This project presents a deep learning pipeline for automated fruit disease detection using a fine-tuned **VGG-16** convolutional neural network. The model classifies fruit images into healthy and diseased categories, enabling early identification of crop diseases to reduce agricultural losses and support sustainable farming.

The work was **peer-reviewed and published at IEEE ICCCIS 2023**.

---

## 📄 Publication

**Detecting Fruit Diseases Using Deep Learning and Image Analysis**
T. Pareek, D. Dobariya, T. Mahajan, D. Bhise
*IEEE International Conference on Connected Computing, Intelligent-Systems and Signal Processing (ICCCIS), 2023*

📎 [View Paper (PDF)](./ICCCIS_Detecting%20Fruit%20Diseases%20using%20Deep%20Learning%20and%20Image%20Analysis.pdf)

---

## 🎯 Objectives

- Automate early-stage fruit disease identification using computer vision
- Evaluate transfer learning with VGG-16 on a multi-class agricultural image dataset
- Apply real-time data augmentation to improve generalization
- Validate model performance using ROC/AUC curves and confusion matrices

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Images | 14,000+ |
| Classes | 5 (healthy + diseased categories) |
| Input Size | 224 × 224 px (VGG-16 standard) |
| Split | Train / Validation |

---

## 🧠 Model Architecture

The model is built on **VGG-16** pretrained on ImageNet, with custom classification layers added on top:

```
Input (224×224×3)
    ↓
VGG-16 Convolutional Base (pretrained, ImageNet weights)
    ↓
Flatten
    ↓
Dense (4096, ReLU) → Dropout
    ↓
Dense (4096, ReLU) → Dropout
    ↓
Dense (5, Softmax)  ← Output: 5 disease classes
```

**Transfer learning strategy:** VGG-16 base layers were frozen during initial training, then selectively unfrozen for fine-tuning.

---

## ⚙️ Methodology

**1. Data Preprocessing**
- Resized all images to 224×224 pixels
- Normalized pixel values to [0, 1]
- Applied real-time data augmentation: horizontal/vertical flips, zoom, rotation, shear

**2. Model Training**
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

**3. Evaluation**
- Validation accuracy and loss curves
- Confusion matrix across all 5 classes
- ROC/AUC curves per class

---

## 📈 Results

| Metric | Score |
|---|---|
| Validation Accuracy | **96 – 98%** |
| Evaluation Method | ROC/AUC + Confusion Matrix |
| Dataset Size | 14,000+ images, 5 classes |

The model significantly outperforms traditional image processing approaches for disease detection, achieving near-human-level accuracy on unseen validation images.

---

## 📂 Repository Structure

```
├── FDD_Final.ipynb                          # Full training pipeline (Jupyter Notebook)
├── ICCCIS_Detecting Fruit Diseases using    # IEEE published paper (PDF)
│   Deep Learning and Image Analysis.pdf
├── Picture1.png                             # Sample result/architecture diagram
├── Picture2.png                             # Sample result
├── Picture3.png                             # Sample result
├── Untitled Diagram.drawio.png              # Model architecture diagram
└── README.md
```

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Model building & training |
| VGG-16 (ImageNet) | Pretrained CNN backbone |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Evaluation metrics (ROC, confusion matrix) |

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/DeepDobariya307/Fruit-Disease-Detection-using-ML.git
cd Fruit-Disease-Detection-using-ML
```

**2. Install dependencies**
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

**3. Open and run the notebook**
```bash
jupyter notebook FDD_Final.ipynb
```

---

## 💼 Real-World Impact

- Enables early detection of crop diseases before visible spread
- Reduces dependency on manual expert inspection
- Scalable to mobile/edge deployment for use by farmers in the field
- Applicable to broader agricultural computer vision use cases

---

## 👤 Author

**Deep Prakashbhai Dobariya**
M.Sc. Artificial Intelligence — BTU Cottbus-Senftenberg
[GitHub](https://github.com/DeepDobariya307) · [LinkedIn](https://www.linkedin.com/in/deep-dobariya-946816221/)

---

## 📜 License

This project is for academic and research purposes.
Dataset sourced from publicly available agricultural image repositories.
