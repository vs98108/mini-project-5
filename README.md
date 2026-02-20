#  Mini Project 5 – CNN Image Classifier  
**COMP-9130 – Applied Artificial Intelligence**  
**Group 9: Aristide Kanamugire & Vibhor Malik**  
**Dataset Option B: Forest Fire, Smoke & Non-Fire Classification**

---

## 📌 Project Overview

This project builds a Convolutional Neural Network (CNN) from scratch to classify forest-related images into three categories:

- 🔥 Fire  
- 🌫 Smoke  
- 🌲 Non-Fire  

Early detection of fire and smoke is critical for preventing large-scale forest fires and reducing environmental damage.

The project compares:

- A Baseline CNN  
- An Improved CNN (with augmentation & regularization)  
- A CNN architecture variation using GlobalAveragePooling2D  

Transfer learning was **not used**, as required.

---

## 👥 Team Contributions

### Aristide Kanamugire
- Dataset setup and Kaggle integration
- Data preprocessing and stratified splitting
- Baseline CNN architecture implementation
- Training pipeline configuration
- Feature map visualization
- Model evaluation metrics (Accuracy, Precision, Recall, F1)
- Report writing and results interpretation
  

### Vibhor Malik
- Data exploration and visualization (class distribution, sample images)
- Improved CNN implementation (augmentation, batch normalization, dropout)
- Confusion matrix generation and analysis
- Misclassification analysis
- CNN architecture variation (GlobalAveragePooling2D)
  

Both members collaborated on:
- Model comparison
- Result analysis
- Final report preparation
- Code testing and debugging
- README documentation and GitHub organization

---

## 📂 Dataset

**Source:** Forest Fire, Smoke & Non-Fire Image Dataset (Kaggle)

Classes:
- Fire  
- Smoke  
- Non-Fire  

### Preprocessing

- Images resized to **128 × 128**
- Pixel values normalized to **[0,1]**
- Stratified split:
  - 70% Training
  - 15% Validation
  - 15% Test

---

##  Model Architectures

### 1️⃣ Baseline CNN

- 3 × (Conv2D + ReLU + MaxPooling)
- Flatten
- Dense (128)
- Softmax output (3 classes)

No augmentation or regularization.

---

### 2️⃣ Improved CNN

Added:

- Data augmentation (rotation, shift, zoom, flip)
- Batch Normalization
- Dropout layers
- EarlyStopping
- ReduceLROnPlateau

Improves generalization and reduces overfitting.

---

### 3️⃣ CNN Architecture Variation (Bonus)

Replaced:

Flatten  

with:

GlobalAveragePooling2D  

This reduces parameter count and improves stability.

---

## ⚙️ Training Setup

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Early stopping used to prevent overfitting

---

## 📈 Results

### Baseline CNN

- Accuracy: 0.9703
- Precision: 0.9704
- Recall: 0.9703
- F1-score: 0.9703

Signs of overfitting observed.

---

### Improved CNN

- Accuracy: 0.9522
- Precision: 0.9529
- Recall: 0.9522
- F1-score: 0.9521

Better validation stability and improved generalization.

---

## 📊 Evaluation Included

- Class distribution visualization
- Sample images
- Accuracy & loss curves
- Confusion matrices
- Model comparison table
- Misclassified images
- Feature map visualization
- Architecture variation experiment

---

## 🧪 How to Run

### Google Colab (Recommended)

1. Upload `kaggle.json`
2. Install Kaggle API
3. Download dataset
4. Run `CNN_Image_Classifier.ipynb`

Dataset path used:

/root/.cache/kagglehub/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset/versions/3

---

### Local Setup

Install:

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

Run:

jupyter notebook CNN_Image_Classifier.ipynb

---

## 📁 Repository Structure

mini-project-5/
│
├── CNN_Image_Classifier.ipynb
└── README.md

---

## 🎯 Learning Outcomes

- Designed CNN from scratch
- Applied augmentation & regularization
- Evaluated using multiple metrics
- Compared architectures
- Analyzed confusion matrices
- Visualized feature maps
- Implemented architecture variation

---

## 📚 References

- Kaggle Dataset  
  https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  

- TensorFlow Documentation  
  https://www.tensorflow.org/api_docs  


---

COMP-9130 – Applied Artificial Intelligence  
Mini Project 5
