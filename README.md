Mini Project 5 – CNN Image Classifier

Course: COMP-9130 – Applied Artificial Intelligence
Group 9: Aristide Kanamugire & Vibhor Malik
Dataset Option B: Forest Fire, Smoke & Non-Fire Classification

🔗 GitHub Repository:
https://github.com/vs98108/mini-project-5

1️⃣ Problem Description & Motivation

Forest fires cause severe environmental, economic, and human damage. Early detection of both fire and smoke is critical for preventing large-scale disasters.

This project develops a Convolutional Neural Network (CNN) from scratch to classify forest-related images into three categories:

🔥 Fire

🌫 Smoke

🌲 Non-Fire

The task is a multi-class image classification problem where the model outputs a probability distribution across three classes and predicts the class with the highest probability.

Transfer learning was intentionally excluded in accordance with the project requirements. The purpose of this constraint was to focus on fundamental CNN design principles, including convolutional feature extraction, pooling strategies, regularization, and generalization behavior. Building the model from scratch allowed direct evaluation of architectural decisions rather than leveraging pretrained weights.

Project Objectives

Design and train a baseline CNN

Improve the model using augmentation and regularization

Evaluate generalization performance using multiple metrics

Compare an architectural variation using GlobalAveragePooling2D

Analyze misclassifications and model behavior

This problem is realistic and challenging because:

Smoke can resemble fog or clouds

Fire can appear small or distant

Lighting conditions vary significantly

2️⃣ Dataset Description

Source:
Kaggle – Forest Fire, Smoke & Non-Fire Image Dataset
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

The dataset is organized into three folders:

Fire/
Smoke/
Non-Fire/
Dataset Characteristics

Images vary in resolution

Classes are slightly imbalanced

Visual similarity exists between Smoke and Non-Fire

Fire intensity and visibility vary across samples

Preprocessing

All images resized to 128 × 128

Pixel values normalized to [0,1]

Stratified split:

70% Training

15% Validation

15% Test

Random seed fixed at 42 for reproducibility

3️⃣ Model Architectures
3.1 Baseline CNN

Architecture:

3 × (Conv2D + ReLU + MaxPooling)

Flatten

Dense (128 neurons)

Softmax output (3 classes)

No augmentation or regularization applied.

Purpose:
Establish a performance baseline.

3.2 Improved CNN

Enhancements added:

Data augmentation:

Rotation

Horizontal flip

Zoom

Width/height shift

Batch Normalization

Dropout layers

EarlyStopping

ReduceLROnPlateau

Purpose:
Reduce overfitting and improve generalization.

3.3 Architecture Variation (Bonus)

Flatten layer replaced with:

GlobalAveragePooling2D

Advantages:

Fewer parameters

Reduced risk of overfitting

More compact feature representation

Improved training stability

4️⃣ Training Configuration

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics:

Accuracy

Precision

Recall

F1-score

Callbacks:

EarlyStopping

ReduceLROnPlateau

5️⃣ Results Summary (Required by Rubric)
🔎 Quantitative Performance
Model	Accuracy	Precision	Recall	F1-score
Baseline CNN	0.9703	0.9704	0.9703	0.9703
Improved CNN	0.9522	0.9529	0.9522	0.9521
Final Model Selection Justification

Although the Baseline CNN achieved higher test accuracy (97.03%), it showed clear overfitting:

Larger gap between training and validation performance

Validation loss less stable

More sensitivity to subtle class variations

The Improved CNN achieved 95.23% accuracy but demonstrated:

Smaller training–validation gap

More stable validation curves

Better separation between Smoke and Non-Fire

Improved robustness to real-world variability

Since real-world fire detection requires strong generalization rather than maximum training accuracy, the Improved CNN is selected as the final model.

6️⃣ Confusion Matrix & Error Analysis
Observed Misclassification Patterns

Thin smoke mistaken for fog/clouds

Small distant flames misclassified as Non-Fire

Bright reflections occasionally mistaken for Fire

The Improved CNN reduced confusion between Smoke and Non-Fire compared to the baseline.

7️⃣ Sample Predictions (Included in Repository)

Example outputs stored in the /images folder:

✅ Correct Fire classification

✅ Correct Smoke classification

✅ Correct Non-Fire classification

❌ Example misclassification

These visual examples allow manual inspection of model behavior.

8️⃣ Setup & Running Instructions (Reproducible)
🔹 Step 1 – Clone Repository
git clone https://github.com/vs98108/mini-project-5.git
cd mini-project-5
🔹 Step 2 – Install Dependencies
pip install -r requirements.txt
🔹 Step 3 – Download Dataset

Go to:
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

Download and extract into:

mini-project-5/data/
├── Fire/
├── Smoke/
└── Non-Fire/
🔹 Step 4 – Run Notebook
jupyter notebook CNN_Image_Classifier.ipynb

Run all cells from top to bottom.

9️⃣ Repository Structure
mini-project-5/
│
├── CNN_Image_Classifier.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── images/
🔟 Dependencies (requirements.txt)
tensorflow
numpy
pandas
matplotlib
scikit-learn
seaborn
1️⃣1️⃣ .gitignore
data/
*.h5
*.ckpt
__pycache__/
.ipynb_checkpoints/
1️⃣2️⃣ Team Member Contributions
Aristide Kanamugire

Dataset preprocessing and stratified splitting

Baseline CNN implementation

Training pipeline configuration

Metric evaluation (Accuracy, Precision, Recall, F1)

Feature map visualization

Report writing and interpretation

Vibhor Malik

Data exploration and visualization

Improved CNN implementation

Augmentation and regularization

Confusion matrix generation

Misclassification analysis

Architecture variation using GlobalAveragePooling2D

Both Members

Model comparison

Debugging

Repository organization

Documentation and README preparation

1️⃣3️⃣ Learning Outcomes

Designed CNN architecture from scratch

Applied augmentation and regularization

Evaluated models using multiple metrics

Performed misclassification analysis

Visualized learned features

Compared architectural variants

Improved model generalization

1️⃣4️⃣ References

Kaggle Dataset:
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

TensorFlow Documentation:
https://www.tensorflow.org/api_docs
