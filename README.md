## 🧠 CNN Image Classification on CIFAR-10 and CIFAR-100

This project explores deep learning techniques for image classification using PyTorch. A custom Convolutional Neural Network (CNN) is first built and trained from scratch on **CIFAR-10**. The learned features from this model are then transferred to **CIFAR-100** to classify a larger set of categories using **transfer learning**.

---

### 📁 Datasets

- **CIFAR-10**: 60,000 32x32 color images across 10 balanced classes
- **CIFAR-100**: 60,000 32x32 color images across 100 fine-grained classes\
  (Both split into 50,000 train and 10,000 test images)

---

### 💠 Project Highlights

- ✅ **Built CNN from scratch for CIFAR-10**

  - 6-level convolutional structure with progressive filters: 16 → 32 → 64 → 128 → 256
  - Experiments with deeper models (starting from 32 filters up to 512)
  - Explore different activation functions: **ReLU**, **LEakyRelu**, **ELU**
  - Use of **Batch Normalization**, and **Dropout** for training stability and regularization
  - MaxPooling layers to downsample and reduce spatial dimensions

- 🔄 **Transfer Learning on CIFAR-100**

  - Used trained CIFAR-10 model weights
  - Retained convolutional layers, fine-tuned classification head for 100 classes
  - Enabled faster convergence and better generalization on the more complex dataset

- 📊 **Training Strategies**

  - Data normalization and augmentation
  - Early stopping based on validation accuracy
  - Accuracy used as the main performance metric (due to balanced datasets)

---

### 📊 Results

| Dataset   | Model Strategy                    | Test Accuracy |
| --------- | --------------------------------- | ------------- |
| CIFAR-10  | Custom CNN (trained from scratch) | *85.6%* |
| CIFAR-100 | Transfer Learning from CIFAR-10   | *67.1%* |

---

### 📦 Libraries

- PyTorch
- torchvision
- torch
- torcham
- numpy
- matplotlib
- scikit-learn
  
---
