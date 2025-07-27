# 🧬 Breast Cancer Classification Using ML & Deep Learning

This repository presents a comprehensive approach to **Breast Cancer Classification** using both traditional **Machine Learning (ML)** algorithms and **Deep Learning (DL)** models. It leverages a real-world dataset to classify tumors as either benign or malignant, aiming to assist in early detection and diagnosis.

---

## 📌 Objective

The goal of this project is to build robust classification models that can accurately predict whether a tumor is benign or malignant using:
- Classical Machine Learning techniques
- Deep Learning with Neural Networks

---

## 📊 Dataset

The dataset used contains various clinical features such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

The target variable is **diagnosis**:
- 0 = Benign
- 1 = Malignant

---

## 🤖 Part 1: Machine Learning Approach

📁 **File:** `Breast Cancer Classification Using ML.ipynb`

### ✅ Steps Followed

1. **Data Preprocessing**
   - Loaded the dataset
   - Checked for missing values and nulls
   - Converted categorical target into numeric format

2. **Exploratory Data Analysis (EDA)**
   - Plotted histograms, pairplots, and heatmaps
   - Studied feature correlation with diagnosis

3. **Feature Scaling**
   - Standardized features using `StandardScaler`

4. **Model Building**
   - Trained the following ML classifiers:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest

5. **Model Evaluation**
   - Classification Report (Accuracy, Precision, Recall, F1-Score)
   - Confusion Matrix
   - ROC-AUC Curve

### 🔍 Results

The best-performing model (**Random Forest**) achieved high accuracy and recall, making it ideal for minimizing false negatives in cancer detection.

---

## 🧠 Part 2: Deep Learning Approach

📁 **File:** `Breast Cancer Classification with NN.ipynb`

### ✅ Steps Followed

1. **Data Preprocessing**
   - Normalized the features
   - Split into training and validation sets

2. **Neural Network Design**
   - Input Layer: Number of neurons = number of features
   - Hidden Layers: Dense layers with ReLU activation
   - Output Layer: Single neuron with Sigmoid activation

3. **Model Compilation**
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam
   - Metric: Accuracy

4. **Training**
   - Trained over several epochs
   - Used EarlyStopping to prevent overfitting
   - Visualized training and validation accuracy/loss

5. **Evaluation**
   - Classification report and confusion matrix
   - ROC Curve and AUC score

### 🔍 Results

The deep learning model showed competitive results, with slightly better performance than classical ML in some metrics.

---

## 📈 Performance Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 96.5%    | 96.0%     | 96.8%  | 96.4%    |
| KNN                  | 94.7%    | 94.3%     | 94.9%  | 94.6%    |
| SVM                  | 97.1%    | 97.3%     | 96.4%  | 96.8%    |
| Decision Tree        | 93.6%    | 93.1%     | 94.2%  | 93.6%    |
| Random Forest        | 98.2%    | 98.3%     | 97.8%  | 98.0%    |
| Neural Network       | 98.3%    | 98.0%     | 98.4%  | 98.2%    |

---

## 📌 Conclusion

This project demonstrates how both Machine Learning and Deep Learning techniques can be effectively applied to medical data for binary classification. While Random Forest performed excellently in the ML pipeline, the Deep Learning model slightly outperformed it in overall evaluation metrics.

---

## 🚀 Future Work

- Hyperparameter tuning (GridSearchCV/RandomizedSearch)
- Model interpretability using SHAP or LIME
- Deployment using Flask or Streamlit
- Cross-validation for better generalization
- Handling class imbalance (if any)

---

## 📬 Contact

For questions, feedback, or collaborations, feel free to reach out via [GitHub](https://github.com/nameisritam) or email.

---

## 🙏 Acknowledgements

- Kaggle / UCI for dataset availability  
- Scikit-learn, Matplotlib, Seaborn, TensorFlow, Keras for tooling and support

