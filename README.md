# Customer Churn Prediction using Random Forest

This project focuses on predicting customer churn using a Random Forest classifier. Churn prediction helps businesses identify customers who are likely to leave, enabling proactive strategies to retain them. In this project, we use the popular **Telco Customer Churn** dataset to train a Random Forest model and evaluate its performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [Next Steps](#next-steps)
- [Conclusion](#conclusion)

## Project Overview
Customer churn is when customers stop doing business with a company. In industries like telecommunications, churn prediction is vital because acquiring new customers costs more than retaining existing ones.

This project builds a churn prediction model using **Random Forest**, a powerful and interpretable machine learning algorithm. The primary objective is to:
- Predict whether a customer will churn or not.
- Identify the most significant factors contributing to customer churn.

## Dataset
The dataset used for this project is the **Telco Customer Churn** dataset,. It contains customer demographics, account information, and service usage features.

### Dataset Features:
- **Demographic Information**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Service Information**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `DeviceProtection`, etc.
- **Account Information**: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable**: `Churn` (Whether the customer churned or not)

### Dataset Link
- Dataset is included with the files

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/customer-churn-prediction-random-forest
   cd customer-churn-prediction-random-forest
   ```

2. Install dependencies:

   The main libraries required are:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `seaborn`
   - `scikit-learn`

## Data Preprocessing
Before building the model, we need to preprocess the data:
- **Remove unnecessary columns**: The `customerID` column is dropped as it's not useful for prediction.
- **Convert `TotalCharges` to numeric**: Some rows in the `TotalCharges` column have missing values which are handled by converting the column to numeric type and dropping the rows with NaN values.
- **Label Encoding**: Categorical variables are encoded using `LabelEncoder` so that they can be used in the machine learning model.
- **Train-test split**: The data is split into training and testing sets (70% training, 30% testing).

## Model Training
We use the **Random Forest Classifier** from `scikit-learn` to build the churn prediction model. The key steps include:
- **Model Initialization**: We initialize the Random Forest classifier with 100 decision trees (`n_estimators=100`).
- **Model Training**: The model is trained on the training set using `fit()`.
- **Prediction**: We use the model to predict churn on the testing set.

## Evaluation
The model is evaluated using:
- **Accuracy**: 78.48%
- **Confusion Matrix**:
  ```
  [[1384  165]
   [ 289  272]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

            0       0.83      0.89      0.86      1549
            1       0.62      0.48      0.55       561

      accuracy                           0.78      2110
     macro avg       0.72      0.69      0.70      2110
  weighted avg       0.77      0.78      0.78      2110
  ```
- **Key Observations**:
  - The model is better at predicting non-churners than churners (as seen from the confusion matrix and recall for Class 1).
  - **Recall for churners (0.48)** is relatively low, which means the model struggles to identify all customers who actually churned.

## Feature Importance
We use the feature importance plot to identify which features contribute the most to churn prediction. The top 10 most important features are:
- `TotalCharges`
- `MonthlyCharges`
- `tenure`
- `Contract`
- `PaymentMethod`
- `OnlineSecurity`
- `TechSupport`
- `gender`
- `OnlineBackup`
- `InternetService`

The plot attached shows the top 10 feature importances based on the Random Forest model.

## Results
- The Random Forest model achieved an accuracy of **78.48%**. 
- The model has a high precision for non-churners but struggles to predict churners (Class 1) accurately.
- The most important features contributing to customer churn are **TotalCharges**, **MonthlyCharges**, and **tenure**.

## Next Steps
To further improve the model, the following steps can be taken:
1. **Address Class Imbalance**: Use techniques like oversampling, undersampling, or SMOTE.
2. **Hyperparameter Tuning**: Perform grid search or random search to optimize Random Forest hyperparameters.
3. **Try Different Algorithms**: Experiment with models like **Gradient Boosting** or **XGBoost** to improve performance.
4. **Feature Engineering**: Create additional features (e.g., interaction terms) to capture more information from the data.

## Conclusion
This project demonstrates the process of building a churn prediction model using Random Forest. The model is effective at predicting non-churners but can be improved for better churn prediction through further tuning and balancing techniques.
