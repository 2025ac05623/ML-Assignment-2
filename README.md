# Machine Learning Assignment 2
## Classification Models and Streamlit Application

## A. Problem Statement

The objective of this assignment is to develop and evaluate multiple machine learning classification models using a common classification dataset.

The models are trained to classify breast cancer cases into two categories: Benign and Malignant. Five different classification algorithms are implemented and compared using multiple evaluation metrics.

An interactive Streamlit application is also developed to allow users to upload test data, select a classification model, generate predictions, and view the corresponding evaluation results.

---

## B. Dataset Description

The dataset used for this assignment is the UCI Breast Cancer Wisconsin dataset.

The dataset contains 569 instances and 30 numerical features related to characteristics of cell nuclei obtained from breast cancer diagnostic images.

The target variable is:

- 0 – Benign
- 1 – Malignant

### Dataset Characteristics

| Property | Value |
|---|---:|
| Dataset | UCI Breast Cancer Wisconsin |
| Number of instances | 569 |
| Number of features | 30 |
| Target variable | Diagnosis |
| Number of classes | 2 |
| Missing values in features | 0 |
| Missing values in target | 0 |
| Training instances | 455 |
| Testing instances | 114 |

The dataset was divided into training and testing datasets. The models were trained using the training dataset and evaluated using the test dataset.

---

## C. GitHub Repository Link

The complete source code, saved models, test data, requirements file, and README are maintained in the GitHub repository.

GitHub Repository:

[GitHub Repository Link]

---

## D. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models were evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## E. Observations on Model Performance

### Logistic Regression

Logistic Regression performed very well on the dataset, achieving an accuracy of 96.49% and an AUC of 0.9960. Its F1 Score of 0.9512 and MCC of 0.9245 indicate strong overall classification performance.

The model provides a strong balance between precision and recall and is one of the best-performing individual models in this comparison.

### Decision Tree

The Decision Tree achieved an accuracy of 92.98%, which was the lowest accuracy among the five models.

Its AUC of 0.9246 and MCC of 0.8492 were also lower than the other models. Although the Decision Tree provides reasonable classification performance, it was not as effective as the ensemble and other models on this dataset.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 95.61% and an AUC of 0.9823.

The model achieved high precision of 0.9744, while its recall was 0.9048. Overall, KNN performed well but was slightly below Logistic Regression and Random Forest.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 93.86% and a high AUC of 0.9934.

The model achieved perfect precision of 1.0000, meaning that the malignant predictions it made were highly precise. However, its recall of 0.8333 was lower than the other top-performing models, resulting in a lower F1 Score of 0.9091.

### Random Forest

Random Forest achieved the highest accuracy of 97.37%.

It also achieved perfect precision of 1.0000, recall of 0.9286, F1 Score of 0.9630, and the highest MCC of 0.9442.

The model provides the strongest overall performance among the models evaluated.

---

## F. Overall Winner

### Random Forest

Random Forest is the overall winner for this dataset.

It achieved:

- Accuracy: 97.37%
- AUC: 0.9929
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC: 0.9442

Random Forest provides the best overall combination of accuracy, precision, recall, F1 Score, and MCC among the evaluated models.

Although Logistic Regression achieved a slightly higher AUC of 0.9960, Random Forest achieved the highest accuracy, F1 Score, and MCC, making it the strongest overall model for this test dataset.

---

## G. Streamlit Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

The application provides the following features:

- CSV test data upload
- Classification model selection
- Prediction and evaluation
- Accuracy, AUC, Precision, Recall, F1 Score and MCC
- Confusion Matrix
- Classification Report

### Application Workflow

1. Upload the test dataset in CSV format.
2. Select a classification model.
3. Generate predictions.
4. View evaluation metrics.
5. View the confusion matrix.
6. View the classification report.

---

## H. Project Structure

```text
ML-Assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic_regression.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
