# Undergraduate Thesis (MTH40A/B): Lending Club 
# By: Paul Desroches (500699067) at Ryerson University

# Objective: To predict whether a loan will default,  
# using a Decision Tree involving specified features.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory
dir = 'C:/Users/pdesr/OneDrive/School/Ryerson University/Undergraduate Thesis/Machine Learning in Business/Lending Club'
os.chdir(dir)

# Import data
excel_file = 'lendingclub_v3.xlsx'
raw = pd.read_excel(excel_file)

# View table head
raw.head()

# Customize/clean data
raw = raw[raw.home_ownership.isin(['OWN','RENT','MORTGAGE','OTHER'])]
raw = raw[raw.verification_status.isin(['Verified','Source Verified'])]
raw = raw[raw.dti < 300]
raw.annual_inc = (raw.annual_inc)/1000
raw.loan_amnt = (raw.loan_amnt)/1000
raw.tot_cur_bal = (raw.tot_cur_bal)/1000
raw = raw.replace({'home_ownership': {'OWN':1, 'MORTGAGE':1, 'RENT':0, 'OTHER':0}})
raw = raw.replace({'loan_status': {'Fully Paid':1, 'Current':1, 'In Grace Period':1, 
                                   'Does not meet the credit policy. Status:Fully Paid':1, 
                                   'Does not meet the credit policy. Status:Charged Off':0, 
                                   'Default':0, 'Charged Off':0, 'Late (16-30 days)':0, 'Late (31-120 days)':0}})
raw = raw.replace({'term': {' 36 months':0, ' 60 months':1}})

# Assign specified features to Python variables
raw = raw.drop(columns=['verification_status'])
raw.columns = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','target']
raw = raw.reset_index(drop=True)

# Split into training set and validation set
train = raw[0:round((len(raw))*0.7)]
val = raw[round((len(raw))*0.7):len(raw)]

data = train

obs = val.shape[0] #number of observations
val = val.reset_index(drop=True)

# set up features and target in train and test sets
X_train = data.drop('target', axis=1)
y_train = data['target']
X_test = val.drop('target', axis=1)
y_test = val['target']

# train model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# test model
target_predict = classifier.predict(X_test)

val['target_predict'] = target_predict

# results
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, target_predict))
print(classification_report(y_test, target_predict))

# Compute results manually:

# True Positive: predict good when actual good
tp = []

for row in list(range(0,obs)):
    if (val.target_predict[row] == 1) & (val.target[row] == 1):
        tp.append(1)
    else:
        tp.append(0)

tp = sum(tp)

# False Positive: predict good when actual bad
fp = []

for row in list(range(0,obs)):
    if (val.target_predict[row] == 1) & (val.target[row] == 0):
        fp.append(1)
    else:
        fp.append(0)

fp = sum(fp)

# False Negative: predict bad when actual good
fn = []

for row in list(range(0,obs)):
    if (val.target_predict[row] == 0) & (val.target[row] == 1):
        fn.append(1)
    else:
        fn.append(0)

fn = sum(fn)

# True Negative: predict bad when actual bad
tn = []

for row in list(range(0,obs)):
    if (val.target_predict[row] == 0) & (val.target[row] == 0):
        tn.append(1)
    else:
        tn.append(0)

tn = sum(tn)

# Compute accuracy, true rates and precision

accuracy = (tp + tn)/obs
true_pos_rate = tp/(tp + fn)
true_neg_rate = tn/(tn + fp)
precision = tp/(tp + fp)

results = {'Accuracy': accuracy, 'True Positive Rate': true_pos_rate, 'True Negative Rate': true_neg_rate, 'Precision': precision}
print(results)





