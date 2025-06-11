#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
#pd.options.display.max_rows = None

data = pd.read_csv("../loan_data.csv")
#print(data) # inspect the features
#print(data.dtypes) # inspect categorical vs numerical features

# Remove outliers
indexAge = data[(data['person_age'] < 10) | (data['person_age'] > 100)].index
data.drop(indexAge, inplace=True)

# Inspect how many categories exist for categorical features
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
fig, axs = plt.subplots(1, len(object_cols), figsize=(12, 5))
for index, col in enumerate(object_cols):
    y = data[col].value_counts()
    sns.barplot(x=list(y.index), y=y, ax=axs[index])
    axs[index].set_title(col)
    axs[index].tick_params(axis='x', labelrotation = 90)
plt.tight_layout()
plt.savefig('categories.pdf')
#plt.show()

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

# Inspect Pearson correlation coefficient rho_XY = cov(X, Y) / (sig_X * sig_Y) for mulit-collinearity
plt.figure(figsize=(11,9))
sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap='bwr', fmt='.2f', linewidths=1, linecolor='k', square=True, annot=True)
plt.tight_layout()
plt.savefig('corr.pdf')
#plt.show()
