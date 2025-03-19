# Concrete Defect Detection Using Deep Learning

## Overview
This project focuses on detecting and classifying concrete defects such as **spalling, honeycombing, and surface voids** using a **hybrid deep learning approach**. The model integrates **EfficientNetB0, VGG16, and transfer learning** for feature extraction, **SVM for void classification**, and **Generative Adversarial Networks (GANs) for data augmentation** to enhance model robustness.

## Features
- **Multi-class Classification**: Detects spalling, honeycombing, and surface voids in concrete structures.
- **Deep Learning-Based Approach**: Utilizes EfficientNetB0, VGG16, and transfer learning to improve feature extraction.
- **GAN for Data Augmentation**: Generates synthetic defect images to handle data imbalance and improve model generalization.
- **Hybrid Model**: Combines CNN for deep feature learning and SVM for void classification.
- **Preprocessing & Optimization**: Includes image augmentation, normalization, hyperparameter tuning, early stopping, and regularization.
- **Achieved Accuracy**: Overall model accuracy of **90.67%** across all defect types.

## Dataset
A custom dataset was collected and preprocessed, consisting of **12,695 images** categorized into:
- **Spalling**: 7,993 images
- **Honeycombing**: 2,367 images
- **Surface Voids**: 1,152 images
- **Non-defective Concrete**: 1,183 images

### Data Preprocessing
- Image resizing to **224×224** pixels
- Data augmentation using **rotation, flipping, contrast adjustment, and noise addition**
- GAN-based synthetic data generation for underrepresented defect categories
- Normalization and feature scaling for stable training

## Model Architecture
1. **EfficientNetB0** and **VGG16** for feature extraction
2. **CNN layers** for additional learning
3. **SVM** for void defect classification
4. **GAN-based synthetic data** to improve model performance
5. **Fully connected layers** with Softmax activation for classification

## Training & Optimization
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score

## Results
| Defect Type      | Accuracy |
|-----------------|----------|
| Spalling        | 80.12%   |
| Honeycombing    | 95.24%   |
| Surface Voids   | 97.66%   |
| **Overall Avg.** | **90.67%** |

## Installation & Usage
### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Scikit-learn
- PyTorch (for GAN implementation)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python train_model.py
```

### Testing on New Images
```bash
python predict.py --image path_to_image.jpg
```

## GAN Implementation
- A **DCGAN (Deep Convolutional GAN)** was used to generate synthetic images for defects with fewer samples.
- The generator network created **realistic defect patterns**, improving classifier robustness.
- The discriminator trained alongside the generator to differentiate between real and fake defect images, enhancing feature diversity.

## Future Improvements
- Implement **attention mechanisms** for better feature localization.
- Enhance GAN architecture for **higher-quality synthetic defect images**.
- Deploy as a **web-based application** for real-time defect analysis.

## Repository Structure
```
├── datasets
│   ├── honeycombing
│   ├── spalling
│   ├── voids
│   ├── plain_concrete
│
├── models
│   ├── efficientnetb0.h5
│   ├── svm_model.pkl
│
├── scripts
│   ├── train_model.py
│   ├── predict.py
│   ├── preprocess.py
│
├── gan_training
│   ├── gan_generator.py
│   ├── gan_discriminator.py
│   ├── train_gan.py
│
├── requirements.txt
├── README.md
```
If you found this project useful, give it a ⭐ on [GitHub](https://github.com/your-repo-link)! 🚀

