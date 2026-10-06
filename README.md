# Transfer Learning on VGG16 for the Classification of Blight Infected Leaves

[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow / Keras](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-red.svg)](https://www.tensorflow.org/)
[![Paper: IEEE ETCCE 2021](https://img.shields.io/badge/Publication-IEEE%20ETCCE%202021-blue.svg)](https://jesminnipu.github.io/Portfolio/#publications)
[![Accuracy: 96.88%](https://img.shields.io/badge/Accuracy-96.88%25-success.svg)](https://jesminnipu.github.io/Portfolio/#projects)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

This repository provides the implementation and experimental workflows for **Transfer Learning** using deep convolutional neural network backbones (**VGG16**, **VGG19**, and **ResNet-50**) to classify foliar blight infections across agricultural plant leaf datasets.

---

## 📑 Associated Peer-Reviewed Publication

> **J. Akther, M. Harun-or-roshid, A. A. Nayan, and M. G. Kibria**  
> *"Transfer learning on VGG16 for the classification of potato leaves infected by blight diseases."*  
> **2021 International Conference on Emerging Technology in Computing, Communication and Electronics (ETCCE)**, IEEE, pp. 1–5, 2021.  
> **Published in:** IEEE Xplore | ETCCE 2021

### 📚 Extended Publications in this Research Track:

1. **J. Akther, M. Harun-or-roshid, and A. A. Nayan**  
   *"Potato Leaves Blight Disease Classification in Bangladesh Perspective Using Convolutional Neural Network (CNN)."*  
   **Engineering Journal**, Vol. 27, No. 7, pp. 27–38, 2023. *(Scopus-indexed)*

2. **A. Khan, J. Akther, and F. Rahman**  
   *"Comparative analysis of potato blight diseases BARI-72 and BARI-73 using a simplified convolutional neural network method."*  
   **International Journal of Advanced Technology and Engineering Exploration (IJATEE)**, Vol. 11, Issue 115, pp. 819–837, 2024.  
   DOI: [10.19101/IJATEE.2024.111100083](https://doi.org/10.19101/IJATEE.2024.111100083)

---

## 🔬 Research Background & Motivation

Foliar blight diseases—such as **Early Blight** (*Alternaria solani*), **Late Blight** (*Phytophthora infestans*), and downy mildew—represent some of the most destructive agricultural pathologies affecting global crop production. Early, automated detection is critical for targeted pesticide intervention and crop preservation.

### Why Transfer Learning?
* **Data Scarcity:** Training deep architectures (16+ layers) from scratch on localized agricultural datasets frequently leads to severe overfitting.
* **Feature Generalization:** Pretrained models on ImageNet capture rich visual hierarchies (edges, corners, textures, complex geometries) that transfer exceptionally well to plant leaf lesion identification.
* **Efficiency:** Freezing convolutional feature extractors significantly accelerates training convergence while preserving high classification accuracy.

---

## 🧠 Model Architectures & Pipeline

The implemented pipeline evaluates pretrained deep neural networks:

1. **VGG16 / VGG19:**
   * Utilizes sequential stacks of $3 \times 3$ convolutional filters with small receptive fields.
   * Convolutional layers frozen to serve as fixed feature representations.
   * Custom dense classification heads attached for multi-class plant pathology.
2. **ResNet-50 Benchmark:**
   * Employs deep residual learning blocks with identity shortcut connections to mitigate vanishing gradient issues.
3. **Data Augmentation & Training:**
   * Dynamic data scaling, shearing, zooming, and horizontal flipping via `ImageDataGenerator`.
   * Optimization with the **Adam** optimizer and `categorical_crossentropy` loss.
   * Evaluated across training and held-out validation partitions.

---

## 💻 Source Code Overview

The complete training, validation, and evaluation pipeline is contained in:

* [`vgg_16_and_vgg_19.py`](./vgg_16_and_vgg_19.py)

### Pipeline Stages:
1. **GPU Configuration:** Configures dynamic GPU memory allocation and growth (`allow_growth = True`).
2. **Backbone Loading:** Ingests pretrained ImageNet weights with `include_top=False`.
3. **Layer Freezing:** Freezes backbone weights (`layer.trainable = False`) to prevent destruction of learned feature detectors.
4. **Custom Top Layers:** Adds `Flatten` followed by `Dense(num_classes, activation='softmax')`.
5. **Data Ingestion:** Streams images from directory trees via `ImageDataGenerator.flow_from_directory`.
6. **Training & Metrics Visualization:** Fits model, plots Training Loss vs. Validation Loss and Training Accuracy vs. Validation Accuracy.
7. **Model Serialization:** Exports trained weights to portable HDF5 format (`model_resnet50.h5`).
8. **Inference:** Preprocesses and evaluates unseen test images.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8, 3.9, or 3.10
* TensorFlow 2.x
* CUDA-enabled GPU (optional, runs on CPU as well)

### Installation & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/JesminNipu/Transfer-learning-on-VGG16-for-the-classification-of-blight-infected-leaves.git
   cd Transfer-learning-on-VGG16-for-the-classification-of-blight-infected-leaves
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```

3. **Organize your dataset:**
   ```text
   dataset/
   ├── Train/
   │   ├── Early_Blight/
   │   ├── Late_Blight/
   │   └── Healthy/
   └── Validation/
       ├── Early_Blight/
       ├── Late_Blight/
       └── Healthy/
   ```

4. **Execute the script:**
   ```bash
   python vgg_16_and_vgg_19.py
   ```

---

## 📂 Repository Structure

```text
Transfer-learning-on-VGG16-for-the-classification-of-blight-infected-leaves/
├── vgg_16_and_vgg_19.py      # Transfer learning model building, training, and inference script
├── README.md                 # Academic and technical research documentation
└── .gitignore                # Model weights (.h5), checkpoints, and Python ignore rules
```

---

## 👩‍💻 Author & Contact

**Jesmin Akther**  
Mobile Application Developer & Computer Science Researcher  
* Department of Information and Communication Engineering (ICE), NSTU  
* **Email:** [jesminnipu1@gmail.com](mailto:jesminnipu1@gmail.com)  
* **GitHub:** [@JesminNipu](https://github.com/JesminNipu)  
* **Academic Portfolio:** [https://jesminnipu.github.io/Portfolio/](https://jesminnipu.github.io/Portfolio/)
