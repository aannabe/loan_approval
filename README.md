# Loan Approval Prediction Models

## The Dataset

Source for `loan_data.csv`:  
https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data

About:  
This loan approval dataset is a collection of financial records and associated information used to determine the eligibility of individuals for obtaining loans from a lending institution.
It includes various factors about the applicant, such as the credit score, income, education, as well as loan factors, such as loan amount, interest rate, and intent for loan.
Similar datasets are commonly used in machine learning and data analysis to develop models and algorithms that predict the likelihood of loan approval based on the given features.

## Models
- Naive Bayes (GaussianNB)
- k-Nearest Neighbor (k-NN)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosted Decision Trees (GBDTs)

## Comparisons
The above models were used as implemented in the `sklearn` package.
The dataset (45k observations) was randomly partitioned into 60/20/20% for training/validation/testing purposes, respectively.
The table below shows the performance of the classifiers in the test partition.

| Classifier Model               | Accuracy | F1 Score | ROC-AUC |
| ------------------------------ | -------- | -------- | ------- |
| Naive Bayes                    | 0.7325   | 0.6237   | 0.9387  |
| k-Nearest Neighbor             | 0.8940   | 0.7453   | 0.9468  |
| Logistic Regression            | 0.8919   | 0.7532   | 0.9477  |
| Support Vector Machine         | 0.9091   | 0.7846   |         |
| Random Forest                  | 0.9275   | 0.8242   | 0.9729  |
| Gradient Boosted Decision Tree | 0.9333   | 0.8419   | 0.9786  |

## Conclusions
Although the above model performances might be specific to this dataset, Random Forest and GBDTs seem to perform the best considering the provided metrics.
Nevertheless, performance variations are expected in real data, in contrast to the performance in the synthetic data as given here.
