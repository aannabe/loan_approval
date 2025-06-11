#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import metrics

pd.options.display.max_columns = None
#pd.options.display.max_rows = None

data = pd.read_csv("../loan_data.csv")
#print(data) # inspect the features
#print(data.dtypes) # inspect categorical vs numerical features

# Remove outliers
indexAge = data[(data['person_age'] < 10) | (data['person_age'] > 100)].index
data.drop(indexAge, inplace=True)

# Encode the categories

# Manually assign the education codes since it is cardinal
data['person_education'] = data['person_education'].replace({
    'High School': 1,
    'Associate': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctorate':5
})

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])
    #data[col] = label_encoder.inverse_transform(data[col])
#print(data)



# Features vs Target
X = data.drop(['loan_status'], axis=1)
Y = data['loan_status']

# Scale the feature space to obtain mean = 0 and std = 1
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.model_selection import train_test_split
train_frac = 0.60
valid_frac = 0.20
test_frac  = 0.20

# Hold out final test set (test_frac %)
X_tmp, X_test, Y_tmp, Y_test = train_test_split(X_scaled, Y, test_size=test_frac, random_state=42, stratify=Y)
# Split the remaining 80 % into train (train_frac %) and valid (valid_frac %)
# train_frac % of the *original* data = train_frac / (train_frac + valid_frac) % of the tmp set
frac_test = 1 - train_frac / (train_frac + valid_frac)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_tmp, Y_tmp, test_size=frac_test, random_state=42, stratify=Y_tmp)

from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression()
#clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
#clf = LogisticRegression(penalty='l2', solver='liblinear', C=0.1)
clf = LogisticRegression(penalty='elasticnet', solver='saga', C=0.1, l1_ratio=0.8, max_iter=500)
clf.fit(X_train, Y_train)
pred_train = clf.predict(X_train)
pred_valid = clf.predict(X_valid)
pred_test  = clf.predict(X_test)

proba_train = clf.predict_proba(X_train)[:, 1]
proba_valid = clf.predict_proba(X_valid)[:, 1]
proba_test  = clf.predict_proba(X_test)[:, 1]

print("SCALED COEFFICIENTS:")
for i, var in enumerate(list(X.columns)):
    print("{:13.9f}: {}".format(clf.coef_[0][i], var))

#print("SCALED COEFFICIENTS:", list(X.columns))
#print(clf.coef_[0])

print("Logistic Regression:")
print("Accuracy score of train set: {:.4f}".format(metrics.accuracy_score(Y_train, pred_train)))
print("Accuracy score of valid set: {:.4f}".format(metrics.accuracy_score(Y_valid, pred_valid)))
print("Accuracy score of test set : {:.4f}".format(metrics.accuracy_score(Y_test,  pred_test)))

print("F1 score of train set: {:.4f}".format(metrics.f1_score(Y_train, pred_train)))
print("F1 score of valid set: {:.4f}".format(metrics.f1_score(Y_valid, pred_valid)))
print("F1 score of test set : {:.4f}".format(metrics.f1_score(Y_test,  pred_test )))

print("ROC-AUC score of train set: {:.4f}".format(metrics.roc_auc_score(Y_train, proba_train)))
print("ROC-AUC score of valid set: {:.4f}".format(metrics.roc_auc_score(Y_valid, proba_valid)))
print("ROC-AUC score of test set:  {:.4f}".format(metrics.roc_auc_score(Y_test,  proba_test )))

