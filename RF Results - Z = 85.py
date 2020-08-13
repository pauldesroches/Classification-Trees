# Undergraduate Thesis (MTH40A/B): Lending Club 
# By: Paul Desroches (500699067) at Ryerson University

# Objective: To predict whether a loan will default,  
# using a Decision Tree involving specified features.

#############################################################
###### Step 0. Import Libraries and Lending Club Data #######
#############################################################

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
#raw = raw.sample(frac=1) #randomize observations

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
#raw = raw.drop(columns=['home_ownership','loan_amnt','verification_status'])
#raw = raw[['annual_inc','term','int_rate','loan_status']]
#raw.columns = ['f1','f2','f3','target']
raw = raw.reset_index(drop=True)

# Split into training set and validation set
train = raw[0:round((len(raw))*0.7)]
val = raw[round((len(raw))*0.7):len(raw)]

# val = raw2
obs = val.shape[0] #number of observations

# Assign specified features to Python variables
val = val.drop(columns=['verification_status'])
val.columns = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','target']
#val = val.sample(frac=1,replace=True)
val = val.reset_index(drop=True)

# Create Strategy - Must choose a z-score which will serve as the threshold for good loan probability

# Add prediction columns to dataframe
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []
p7 = []
p8 = []
p9 = []
p10 = []
p11 = []
p12 = []
p13 = []
p14 = []
p15 = []
p16 = []
p17 = []
p18 = []
p19 = []
p20 = []
p21 = []

# Add final prediction column to dataframe
target_predict = []

# p1
for row in list(range(0,obs)):
    if ((val.f6[row] == 1) & (val.f9[row] <= 232.24) & (val.f5[row] <= 35)):
        p1.append(0)
    else:
        p1.append(1)
        
val['p1'] = p1

# p2
for row in list(range(0,obs)):
    if ((val.f3[row] <= 680) & (val.f8[row] > 5) & (val.f9[row] <= 128.78)):
        p2.append(0)
    else:
        p2.append(1)
        
val['p2'] = p2

# p3
for row in list(range(0,obs)):
    if ((val.f7[row] > 12.12) & (val.f8[row] > 3)):
        p3.append(0)
    else:
        p3.append(1)
        
val['p3'] = p3

# p4
for row in list(range(0,obs)):
    if ((val.f2[row] > 20.1) & (val.f1[row] == 1) & (val.f4[row] <= 95)) or ((val.f2[row] > 20.1) & (val.f1[row] == 0)):
        p4.append(0)
    else:
        p4.append(1)
        
val['p4'] = p4

# p5
for row in list(range(0,obs)):
    if ((val.f2[row] > 21.3) & (val.f5[row] > 1.6)):
        p5.append(0)
    else:
        p5.append(1)
        
val['p5'] = p5

# p6
for row in list(range(0,obs)):
    if ((val.f2[row] > 19.80) & (val.f1[row] == 0) & (val.f8[row] > 9)):
        p6.append(0)
    else:
        p6.append(1)
        
val['p6'] = p6

# p7
for row in list(range(0,obs)):
    if ((val.f6[row] == 1) & (val.f9[row] <= 213.07) & (val.f2[row] > 11.22)) or ((val.f6[row] == 0) & (val.f2[row] > 28.2)):
        p7.append(0)
    else:
        p7.append(1)
        
val['p7'] = p7

# p8
for row in list(range(0,obs)):
    if ((val.f7[row] > 16.02)):
        p8.append(0)
    else:
        p8.append(1)
        
val['p8'] = p8

# p9
for row in list(range(0,obs)):
    if ((val.f7[row] > 12.02)):
        p9.append(0)
    else:
        p9.append(1)
        
val['p9'] = p9

# p10
for row in list(range(0,obs)):
    if ((val.f2[row] > 21.4) & (val.f9[row] <= 423.36)):
        p10.append(0)
    else:
        p10.append(1)
        
val['p10'] = p10

# p11
for row in list(range(0,obs)):
    if ((val.f7[row] > 14.32)):
        p11.append(0)
    else:
        p11.append(1)
        
val['p11'] = p11

# p12
for row in list(range(0,obs)):
    if ((val.f7[row] > 12.92) & (val.f5[row] > 1.9) & (val.f3[row] <= 770)):
        p12.append(0)
    else:
        p12.append(1)
        
val['p12'] = p12

# p13
for row in list(range(0,obs)):
    if ((val.f2[row] > 19.8) & (val.f4[row] > 80) & (val.f8[row] > 26)) or ((val.f2[row] > 19.8) & (val.f4[row] <= 80)):
        p13.append(0)
    else:
        p13.append(1)
        
val['p13'] = p13

# p14
for row in list(range(0,obs)):
    if ((val.f7[row] > 12.92) & (val.f3[row] > 690)) or ((val.f7[row] > 12.92) & (val.f3[row] <= 690) & (val.f4[row] <= 144)):
        p14.append(0)
    else:
        p14.append(1)
        
val['p14'] = p14

# p15
for row in list(range(0,obs)):
    if ((val.f3[row] > 695) & (val.f5[row] <= 10.9) & (val.f9[row] <= 4)) or ((val.f3[row] <= 695) & (val.f5[row] > 10.3) & (val.f9[row] <= 174.2)):
        p15.append(0)
    else:
        p15.append(1)
        
val['p15'] = p15

# p16
for row in list(range(0,obs)):
    if ((val.f6[row] == 1) & (val.f4[row] <= 104.9) & (val.f8[row] > 4)) or ((val.f6[row] == 0) & (val.f4[row] <= 49) & (val.f8[row] > 6)):
        p16.append(0)
    else:
        p16.append(1)
        
val['p16'] = p16

# p17
for row in list(range(0,obs)):
    if ((val.f3[row] > 695) & (val.f6[row] == 1) & (val.f2[row] <= 6.98)) or ((val.f3[row] > 695) & (val.f6[row] == 0) & (val.f2[row] <= 1.4)) or ((val.f3[row] <= 695) & (val.f2[row] > 21.22)) or ((val.f3[row] <= 695) & (val.f2[row] <= 21.22) & (val.f6[row] == 1)):
        p17.append(0)
    else:
        p17.append(1)
        
val['p17'] = p17

# p18
for row in list(range(0,obs)):
    if ((val.f6[row] == 1) & (val.f9[row] <= 374)):
        p18.append(0)
    else:
        p18.append(1)
        
val['p18'] = p18

# p19
for row in list(range(0,obs)):
    if ((val.f2[row] > 14.5) & (val.f5[row] > 9)) or ((val.f2[row] > 14.5) & (val.f5[row] <= 9) & (val.f4[row] <= 32)):
        p19.append(0)
    else:
        p19.append(1)
        
val['p19'] = p19

# p20
for row in list(range(0,obs)):
    if ((val.f9[row] <= 234) & (val.f8[row] > 9)):
        p20.append(0)
    else:
        p20.append(1)
        
val['p20'] = p20

# p21
for row in list(range(0,obs)):
    if ((val.f2[row] > 21.3) & (val.f6[row] == 1)) or ((val.f2[row] > 21.3) & (val.f6[row] == 0) & (val.f1[row] == 0)) or ((val.f2[row] <= 21.3) & (val.f6[row] == 1) & (val.f1[row] == 0)):
        p21.append(0)
    else:
        p21.append(1)
        
val['p21'] = p21

# FINAL PREDICTION
for row in list(range(0,obs)):
    if sum(val.loc[row,['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p17','p18','p19','p20','p21']]) <= 10:
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