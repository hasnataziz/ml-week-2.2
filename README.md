# ml-week-2.2
# Loan Prediction Model: Decision Trees and Random Forests

This repository contains code for implementing and comparing decision trees and random forests for predicting loan approvals using the Loan Prediction dataset from Kaggle.

## Objective

The objective is to:
1. Implement a decision tree model and visualize the tree.
2. Implement a random forest model to improve performance and reduce overfitting.
3. Implement a randomized decision tree.
4. Compare the performance of the decision tree, random forest, and randomized decision tree using accuracy, F1-score, and other relevant metrics.

## Dataset

The dataset used is the Loan Prediction dataset from Kaggle, which involves predicting loan approval based on attributes like gender, marital status, income, and loan amount.

[Loan Prediction Dataset on Kaggle](https://www.kaggle.com/code/bhavikbb/loan-prediction-dataset/output)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/loan-prediction.git
   cd loan-prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure the dataset is available in the specified path. Update the path in the script if necessary:

python
Copy code
loan = pd.read_csv('/path/to/your/dataset.csv')
Code Explanation
Step-by-Step Code Walkthrough
Import Libraries: Import necessary libraries for data manipulation, preprocessing, model building, evaluation, and visualization.

python
Copy code
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
Load Dataset: Load the Loan Prediction dataset from a specified path and display its basic information and first few rows to understand the structure of the data.

python
Copy code
loan = pd.read_csv('/path/to/your/dataset.csv')
print("Initial Data Info:")
print(loan.info())
print(loan.head())
Handle Missing Values:

Fill missing values for categorical columns with the most frequent value.
Fill missing values for numerical columns with the median value.
python
Copy code
categorical_cols = ['Gender', 'Dependents', 'Self_Employed', 'Credit_History']
imputer = SimpleImputer(strategy='most_frequent')
loan[categorical_cols] = imputer.fit_transform(loan[categorical_cols])

numerical_cols = ['LoanAmount', 'Loan_Amount_Term']
imputer = SimpleImputer(strategy='median')
loan[numerical_cols] = imputer.fit_transform(loan[numerical_cols])
Encode Categorical Variables: Encode all categorical variables to numerical values using LabelEncoder.

python
Copy code
label_encoders = {}
for column in loan.select_dtypes(include=['object']).columns:
    if column != 'Loan_ID':
        label_encoders[column] = LabelEncoder()
        loan[column] = label_encoders[column].fit_transform(loan[column])
Display Processed Data: Show the basic information and first few rows of the processed data.

python
Copy code
print("\nProcessed Data Info:")
print(loan.info())
print(loan.head())
Feature and Target Separation: Separate the features (X) and the target variable (y).

python
Copy code
X = loan.drop(columns=['Loan_ID', 'Credit_History'])
y = loan['Credit_History']
Train-Test Split: Split the data into training and testing sets.

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Train Decision Tree: Initialize and train a DecisionTreeClassifier.

python
Copy code
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
Visualize Decision Tree: Visualize the trained decision tree using plot_tree.

python
Copy code
plt.figure(figsize=(20,10))
plot_tree(decision_tree, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()
Train Random Forest: Initialize and train a RandomForestClassifier.

python
Copy code
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
Train Randomized Decision Tree: Initialize and train a RandomForestClassifier with only one estimator and bootstrap disabled.

python
Copy code
random_tree = RandomForestClassifier(n_estimators=1, bootstrap=False, random_state=42)
random_tree.fit(X_train, y_train)

randomized_decision_tree = random_tree.estimators_[0]
Visualize Randomized Decision Tree: Visualize the extracted randomized decision tree using plot_tree.

python
Copy code
plt.figure(figsize=(20,10))
plot_tree(randomized_decision_tree, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()
Make Predictions: Use the trained models to make predictions on the test data.

python
Copy code
y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = random_forest.predict(X_test)
y_pred_random_tree = random_tree.predict(X_test)
Evaluate Performance: Calculate evaluation metrics for each model's predictions and store them in dictionaries.

python
Copy code
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score
}

performance_tree = {name: metric(y_test, y_pred_tree) for name, metric in metrics.items()}
performance_forest = {name: metric(y_test, y_pred_forest) for name, metric in metrics.items()}
performance_random_tree = {name: metric(y_test, y_pred_random_tree) for name, metric in metrics.items()}
Generate Classification Reports: Generate detailed classification reports for each model.

python
Copy code
report_tree = classification_report(y_test, y_pred_tree, target_names=['0', '1'])
report_forest = classification_report(y_test, y_pred_forest, target_names=['0', '1'])
report_random_tree = classification_report(y_test, y_pred_random_tree, target_names=['0', '1'])
Display Performance Comparison: Print the evaluation metrics and classification reports for each model to compare their performance.

python
Copy code
print("\nDecision Tree Performance:")
for metric, value in performance_tree.items():
    print(f"{metric}: {value:.4f}")

print("\nRandom Forest Performance:")
for metric, value in performance_forest.items():
    print(f"{metric}: {value:.4f}")

print("\nRandomized Decision Tree Performance:")
for metric, value in performance_random_tree.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report for Decision Tree:\n", report_tree)
print("Classification Report for Random Forest:\n", report_forest)
print("Classification Report for Randomized Decision Tree:\n", report_random_tree)
Expected Output
A Jupyter notebook with implementations and performance comparisons.
Visualizations of tree diagrams and performance metrics.
Detailed documentation of the decisions made during the model building process.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes.
