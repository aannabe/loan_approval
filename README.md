# Loan Approval Prediction Models

## The Dataset

Source for `loan_data.csv`:  
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data

About:  
This loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals for obtaining loans from a lending institution.
It includes various factors about the applicant, such as the credit score, income, education, as well as loan factors, such as loan amount, interest rate, and intent for loan.
Similar datasets are commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.

## Models
- Logistic Regression
- Random Forest
- k-Nearest Neighbor (k-NN)
- Support Vector Machine (SVM)

## Comparisons
The above models were used as implemented in the `sklearn` package.
The table below shows the performance of the classifiers.
The dataset (45k observations) was randomly partitioned into 60/20/20% for training/validation/testing purposes.

| Classifier Model       | Accuracy | F1 Score | ROC-AUC |
| ---------------------- | -------- | -------- | ------- |
| k-Nearest Neighbor     | 0.8940   | 0.7453   | 0.9468  |
| Logistic Regression    | 0.8919   | 0.7532   | 0.9477  |
| Random Forest          | 0.9275   | 0.8242   | 0.9729  |
| Support Vector Machine |  |  |  |

