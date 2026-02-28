#  **Mini Project 5 – CNN Image Classifier**

**Course:** COMP-9130 – Applied Artificial Intelligence  
**Group 9:** Aristide Kanamugire & Vibhor Malik  
**Dataset Option B:** Forest Fire, Smoke & Non-Fire Classification  

🔗 **GitHub Repository:**  
https://github.com/vs98108/mini-project-5  

---

# 📌 **1. Problem Description & Motivation**

Forest fires cause severe environmental destruction, economic loss, and risk to human life. Early detection of both **fire** and **smoke** is essential for preventing large-scale disasters.

This project develops a **Convolutional Neural Network (CNN) from scratch** to classify forest-related images into three categories:

- 🔥 **Fire**  
- 🌫 **Smoke**  
- 🌲 **Non-Fire**  

The task is a **multi-class image classification problem**, where the model outputs a probability distribution across three classes and predicts the class with the highest probability.

---

## 🎯 **Project Objectives**

- Design a baseline CNN architecture  
- Improve performance using augmentation and regularization  
- Compare generalization performance across models  
- Evaluate an architectural variation using GlobalAveragePooling2D  
- Analyze misclassifications and model behavior  

---

##  **Transfer Learning Constraint**

Transfer learning was intentionally excluded in accordance with the project requirements. The purpose of this constraint was to focus on fundamental CNN design principles, including convolutional feature extraction, pooling strategies, regularization, and generalization behavior. Building the model from scratch allowed direct evaluation of architectural decisions rather than leveraging pretrained weights.

Although transfer learning is widely used in industry for computer vision tasks, this project emphasizes architectural understanding and core deep learning concepts.

---

# 📂 **2. Dataset Description**

**Source:**  
Kaggle – Forest Fire, Smoke & Non-Fire Image Dataset  
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  

The dataset contains three folders:

**Fire/**  
**Smoke/**  
**Non-Fire/**  

---

## 📊 **Dataset Characteristics**

- Images vary in resolution  
- Classes are slightly imbalanced  
- Smoke and Non-Fire images can be visually similar  
- Fire may appear small or distant  
- Lighting and environmental conditions vary significantly  

---

##  **Preprocessing Steps**

- Images resized to **128 × 128**
- Pixel values normalized to **[0, 1]**
- Stratified data split:
  - 70% Training  
  - 15% Validation  
  - 15% Test  
- Random seed fixed at **42** for reproducibility  

---

#  **3. Model Architectures**

## 🔹 **3.1 Baseline CNN**

Architecture:

- 3 × (Conv2D + ReLU + MaxPooling)
- Flatten
- Dense (128 neurons)
- Softmax output layer (3 classes)

Purpose: Establish a reference performance benchmark without augmentation or regularization.

---

## 🔹 **3.2 Improved CNN**

Enhancements added:

- Data augmentation:
  - Rotation
  - Horizontal flip
  - Zoom
  - Width/height shift
- Batch Normalization
- Dropout layers
- EarlyStopping
- ReduceLROnPlateau

Purpose: Reduce overfitting and improve generalization to unseen images.

---

## 🔹 **3.3 Architecture Variation (Bonus)**

Flatten layer replaced with:

**GlobalAveragePooling2D**

Advantages:

- Fewer trainable parameters  
- Reduced risk of overfitting  
- More compact spatial feature representation  
- Improved training stability  

---

# ⚙️ **4. Training Configuration**

- Optimizer: **Adam**  
- Loss Function: **Categorical Crossentropy**  
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score  
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau  

---

# 📊 **5. Results Summary**

## 🔎 **Quantitative Model Performance**

| Model | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Baseline CNN | 0.9703 | 0.9704 | 0.9703 | 0.9703 |
| Improved CNN | 0.9522 | 0.9529 | 0.9522 | 0.9521 |

---

##  **Final Model Selection Justification**

Although the **Baseline CNN achieved higher raw test accuracy (97.03%)**, it demonstrated clear overfitting:

- Larger training–validation performance gap  
- Less stable validation loss  
- Higher sensitivity to subtle class similarities  

The **Improved CNN achieved 95.23% accuracy**, but demonstrated:

- Smaller train–validation gap  
- More stable validation curves  
- Reduced confusion between Smoke and Non-Fire  
- Better robustness to image variability  

Since real-world fire detection systems prioritize reliability and generalization over peak training accuracy, the **Improved CNN is selected as the final model**.

---

# 📉 **6. Confusion Matrix & Error Analysis**

### 🔍 **Observed Misclassification Patterns**

- Thin smoke misclassified as fog/clouds  
- Small distant flames misclassified as Non-Fire  
- Bright reflections occasionally mistaken for Fire  

The improved model reduced confusion between Smoke and Non-Fire compared to the baseline.

---

# **7. Sample Predictions**

Example outputs stored in the `/images` folder include:

- Correct Fire prediction  
- Correct Smoke prediction  
- Correct Non-Fire prediction  
- Example misclassification  

These examples allow visual inspection of learned feature behavior.

---

# 🧪 **8. Setup & Running Instructions**

## 🔹 **Step 1 – Clone Repository**

git clone https://github.com/vs98108/mini-project-5.git  
cd mini-project-5  

---

## 🔹 **Step 2 – Install Dependencies**

pip install -r requirements.txt  

---

## 🔹 **Step 3 – Download Dataset**

1. Visit:  
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  

2. Extract dataset into:

mini-project-5/data/  
├── Fire/  
├── Smoke/  
└── Non-Fire/  

---

## 🔹 **Step 4 – Run Notebook**

jupyter notebook CNN_Image_Classifier.ipynb  

Run all cells sequentially.

---

#  **9. Repository Structure**

mini-project-5/  
│  
├── CNN_Image_Classifier.ipynb  
├── requirements.txt  
├── README.md  
├── .gitignore  
└── images/  

---

# 📋 **10. Dependencies (requirements.txt)**

tensorflow  
numpy  
pandas  
matplotlib  
scikit-learn  
seaborn  

---

# **11. .gitignore**

data/  
*.h5  
*.ckpt  
__pycache__/  
.ipynb_checkpoints/  

---

# 👥 **12. Team Member Contributions**

### **Aristide Kanamugire**

- Dataset preprocessing and stratified splitting  
- Baseline CNN implementation  
- Training pipeline configuration  
- Metric evaluation (Accuracy, Precision, Recall, F1)  
- Feature map visualization  
- Report writing and interpretation  

### **Vibhor Malik**

- Data exploration and visualization  
- Improved CNN implementation  
- Data augmentation and regularization  
- Confusion matrix generation  
- Misclassification analysis  
- Architecture variation using GlobalAveragePooling2D  

### **Both Members**

- Model comparison  
- Debugging  
- Repository organization  
- Documentation and README preparation  

---

#  **13. Learning Outcomes**

- Designed CNN architecture from scratch  
- Applied augmentation and regularization techniques  
- Evaluated models using multiple performance metrics  
- Performed misclassification analysis  
- Visualized learned convolutional features  
- Compared architectural variants  
- Improved generalization performance  

---

# 📚 **14. References**

Kaggle Dataset:  
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  

TensorFlow Documentation:  
https://www.tensorflow.org/api_docs  
