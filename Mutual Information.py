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
raw = raw.drop(columns=['verification_status'])
#raw = raw[['annual_inc','term','int_rate','loan_status']]
raw.columns = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','target']
raw = raw.reset_index(drop=True)

# Split into training set and validation set
train = raw[0:round((len(raw))*0.7)]
val = raw[round((len(raw))*0.7):len(raw)]

data = train

def IGL(frame,X):
    threshold = (min(frame[X])+max(frame[X]))/2
    pos_good = (frame[(frame[X] == 1) & (frame.target == 1)].count()[0])/obs
    pos_default = (frame[(frame[X] == 1) & (frame.target == 0)].count()[0])/obs
    neg_good = (frame[(frame[X] == 0) & (frame.target == 1)].count()[0])/obs
    neg_default = (frame[(frame[X] == 0) & (frame.target == 0)].count()[0])/obs
    prob_pos = pos_good + pos_default
    prob_neg = neg_good + neg_default
    # Compute conditional probabilities needed for information gain
    prob_good_if_pos = pos_good/prob_pos
    prob_default_if_pos = pos_default/prob_pos
    prob_good_if_neg = neg_good/prob_neg
    prob_default_if_neg = neg_default/prob_neg
    # Compute expected entropy given new information
    expected_entropy = -1*(prob_pos*(prob_good_if_pos*np.log(prob_good_if_pos)+
                                     prob_default_if_pos*np.log(prob_default_if_pos))+
                           prob_neg*(prob_good_if_neg*np.log(prob_good_if_neg)+
                                     prob_default_if_neg*np.log(prob_default_if_neg)))
    # Compute information gain from first feature (home ownership)
    info_gain_X = entropy - expected_entropy
    #print("Information Gain is: ",info_gain_X)
    return [info_gain_X,threshold]

# Let this function be defined as the Information Gain (IG) of numerical feature X:
def IGN(frame,X,a,b,h):
    thrX = np.arange(a,b,h) # set of thresholds 
    exp_entropy_set = [] #storage for entropy values
    for threshold in thrX:
        pos_good = (frame[(frame[X] > threshold) & (frame.target == 1)].count()[0])/obs
        pos_default = (frame[(frame[X] > threshold) & (frame.target == 0)].count()[0])/obs
        neg_good = (frame[(frame[X] <= threshold) & (frame.target == 1)].count()[0])/obs
        neg_default = (frame[(frame[X] <= threshold) & (frame.target == 0)].count()[0])/obs
        prob_pos = pos_good + pos_default
        prob_neg = neg_good + neg_default
        prob_good_if_pos = pos_good/prob_pos
        prob_default_if_pos = pos_default/prob_pos
        prob_good_if_neg = neg_good/prob_neg
        prob_default_if_neg = neg_default/prob_neg
        exp_entropy_set.append(-1*(prob_pos*(prob_good_if_pos*np.log(prob_good_if_pos)+
                                             prob_default_if_pos*np.log(prob_default_if_pos))+
                                   prob_neg*(prob_good_if_neg*np.log(prob_good_if_neg)+
                                             prob_default_if_neg*np.log(prob_default_if_neg))))
    #plt.plot(thrX,exp_entropy_set) #plot the expected entropy over threshold values
    threshX = thrX[exp_entropy_set==np.nanmin(exp_entropy_set)][0]
    #print("Threshold is: ",threshX)
    expected_entropy = np.nanmin(exp_entropy_set)
    #print("Expected Entropy is: ",expected_entropy)
    info_gain_X = entropy - expected_entropy
    #print("Information Gain is: ",info_gain_X)
    return [info_gain_X,threshX]


# Test out IG for all features #

#First, we need to evaluate entropy given the current situation
obs = data.shape[0] #number of observations
outcomes = data['target'].value_counts()
good = (outcomes[1])/obs #good loans percentage
default = 1 - good #defaulted loans percentage
entropy = -1*((good*np.log(good))+(default*np.log(default)))

# Determine IG for each feature
info_gain_f1 = IGL(data,'f1')
info_gain_f2 = IGN(data,'f2',min(data.f2)+0.0001,max(data.f2),0.01) #numerical
info_gain_f3 = IGN(data,'f3',min(data.f3)+0.0001,max(data.f3),1) #numerical
info_gain_f4 = IGN(data,'f4',min(data.f4)+0.0001,max(data.f4),0.1) #numerical
info_gain_f5 = IGN(data,'f5',min(data.f5)+0.0001,max(data.f5),0.01) #numerical
info_gain_f6 = IGL(data,'f6')
info_gain_f7 = IGN(data,'f7',min(data.f7)+0.0001,max(data.f7),0.01) #numerical
info_gain_f8 = IGN(data,'f8',min(data.f8)+0.0001,max(data.f8),1) #numerical
info_gain_f9 = IGN(data,'f9',min(data.f9)+0.0001,max(data.f9),0.1) #numerical