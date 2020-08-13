# Undergraduate Thesis (MTH40A/B): Lending Club 
# By: Paul Desroches (500699067) at Ryerson University

# Objective: To predict whether a loan will default,  
# using a Decision Tree involving specified features.

# Previous script computes classification tree.
# This script applies a specified strategy to observe results.

# Import libraries and val data

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

obs = val.shape[0] #number of observations

# Assign specified features to Python variables
val.columns = ['f1','f2','f3','f4','target']
val = val.reset_index(drop=True)

# Create Strategy - Must choose a z-score which will serve as the threshold for good loan probability

# Add loan prediction column to dataframe
target_predict = []

for row in list(range(0,obs)):
    if ((val.f4[row] > 13.62)) or ((val.f4[row] <= 13.62) & (val.f2[row] <= 715) & (val.f1[row] > 30.85)):
        target_predict.append(0)
    else:
        target_predict.append(1)
        
val['target_predict'] = target_predict

# Compute Results:
    
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

del(accuracy, fn, fp, obs, precision, results, row, tn, tp, true_neg_rate, true_pos_rate, val, target_predict)
