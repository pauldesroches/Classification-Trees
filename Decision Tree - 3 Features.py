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
#raw = raw.sample(frac=1,replace=True) #randomize observations

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
raw = raw[['fico_range_low','term','int_rate','loan_status']]
raw.columns = ['f1','f2','f3','target']
raw = raw.reset_index(drop=True)

# Split into training set and validation set
train = raw[0:round((len(raw))*0.7)]
val = raw[round((len(raw))*0.7):len(raw)]

# Sample with replacement the training set from the original training set
#train = train.sample(frac=1,replace=True)
train = train.reset_index(drop=True)

data = train

##################################################
###### Step 1. Standardized Functions ############
##################################################

# Let this function be defined as the Information Gain (IG) of logical (1 or 0) feature X:

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
    expected_entropy = -1*(prob_pos*(prob_good_if_pos*np.log(prob_good_if_pos)+prob_default_if_pos*np.log(prob_default_if_pos))+prob_neg*(prob_good_if_neg*np.log(prob_good_if_neg)+prob_default_if_neg*np.log(prob_default_if_neg)))
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
        exp_entropy_set.append(-1*(prob_pos*(prob_good_if_pos*np.log(prob_good_if_pos)+prob_default_if_pos*np.log(prob_default_if_pos))+prob_neg*(prob_good_if_neg*np.log(prob_good_if_neg)+prob_default_if_neg*np.log(prob_default_if_neg))))
    #plt.plot(thrX,exp_entropy_set) #plot the expected entropy over threshold values
    threshX = thrX[exp_entropy_set==np.nanmin(exp_entropy_set)][0]
    #print("Threshold is: ",threshX)
    expected_entropy = np.nanmin(exp_entropy_set)
    #print("Expected Entropy is: ",expected_entropy)
    info_gain_X = entropy - expected_entropy
    #print("Information Gain is: ",info_gain_X)
    return [info_gain_X,threshX]

# #Plot expected entropy
# fig = plt.figure()
# plt.plot(thrX,exp_entropy_set)
# #fig.suptitle('test title', fontsize=20)
# plt.xlabel('FICO Threshold', fontsize=12)
# plt.ylabel('Expected Entropy', fontsize=12)
# #fig.savefig('dti.png')

# Let this function determine the feature and its threshold at each node:
def node(prev,curr):
        # Entropy given current situation
        obs = data.shape[0] #number of observations
        outcomes = data['target'].value_counts()
        good = (outcomes[1])/obs #good loans percentage
        default = 1 - good #defaulted loans percentage
        entropy = -1*((good*np.log(good))+(default*np.log(default)))
        if prev[-1] in prune:
            print('This decision point does not exist. Node:',curr)
            prune.append(curr)
            tree.append(['NA','NA',good])
        else:
            if obs <= 1000:
                print('Prune this decision point. Probability of a good loan is: ', good)
                prune.append(curr)
                tree.append(['End','NA',good])
            else:
                # IG for each feature (except those in existing nodes)
                info_gain_f1 = [0,0]
                info_gain_f2 = [0,0]
                info_gain_f3 = [0,0]
                for i in prev:
                    if 'f1' not in tree[i][0]:
                        info_gain_f1 = IGN(data,'f1',min(data.f1)+0.0001,max(data.f1),1) #numerical
                    else:
                        info_gain_f1 = [0,0]
                        break
                for i in prev:
                    if 'f2' not in tree[i][0]:
                        info_gain_f2 = IGL(data,'f2')
                    else:
                        info_gain_f2 = [0,0]
                        break
                for i in prev:
                    if 'f3' not in tree[i][0]:
                        info_gain_f3 = IGN(data,'f3',min(data.f3)+0.0001,max(data.f3),0.1) #numerical
                    else:
                        info_gain_f3 = [0,0]
                        break
                # We may now determine the root node by taking the highest information gain
                dic = {info_gain_f1[0]:"f1",info_gain_f2[0]:"f2",info_gain_f3[0]:"f3"}
                node = dic.get(max(dic))
                dic2 = {'f1':info_gain_f1[1],'f2':info_gain_f2[1],'f3':info_gain_f3[1]}
                threshold = dic2.get(node)
                tree.append([node,threshold,good])
                print("The Node is: ",node)
                print("The Threshold is: ",threshold)

##################################################
###### Step 2. Determination of Root Node ########
##################################################

#First, we need to evaluate entropy given the current situation
obs = data.shape[0] #number of observations
outcomes = data['target'].value_counts()
good = (outcomes[1])/obs #good loans percentage
default = 1 - good #defaulted loans percentage
entropy = -1*((good*np.log(good))+(default*np.log(default)))
tree = [] #create storage for tree thresholds
prune = [] #create storage for pruned nodes

# Determine IG for each feature
info_gain_f1 = IGN(data,'f1',min(data.f1)+0.0001,max(data.f1),1) #numerical
info_gain_f2 = IGL(data,'f2')
info_gain_f3 = IGN(data,'f3',min(data.f3)+0.0001,max(data.f3),0.1) #numerical

# We may now determine the root node by taking the highest information gain
dic = {info_gain_f1[0]:"f1",info_gain_f2[0]:"f2",info_gain_f3[0]:"f3"}
root = dic.get(max(dic))
dic2 = {'f1':info_gain_f1[1],'f2':info_gain_f2[1],'f3':info_gain_f3[1]}
threshold = dic2.get(root)
tree.append([root,threshold,good])
print("The Root Node is: ",root)
print("The Threshold is: ",threshold)

####################################################################
###### Step 3. Determine feature and threshold at each node ########
####################################################################

# Node 1
# Filter dataframe to current situation
prev = [0]
curr = 1
try: 
    data = data[data[root] > dic2.get(root)]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 2
# Filter dataframe to current situation
prev = [0]
curr = 2
try:
    data = data[data[root] <= dic2.get(root)]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 3
prev = [0,1]
curr = 3
try:
    data = data[data[root] > dic2.get(root)]
    data = data[data[tree[1][0]] > tree[1][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 4
prev = [0,1]
curr = 4
try:
    data = data[data[root] > dic2.get(root)]
    data = data[data[tree[1][0]] <= tree[1][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 5
prev = [0,2]
curr = 5
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] > tree[2][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 6
prev = [0,2]
curr = 6
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] <= tree[2][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 7
prev = [0,1,3]
curr = 7
try:
    data = data[data[root] > dic2.get(root)]
    data = data[data[tree[1][0]] > tree[1][1]]
    data = data[data[tree[3][0]] > tree[3][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 8
prev = [0,1,3]
curr = 8
try:
    data = data[data[root] > dic2.get(root)]
    data = data[data[tree[1][0]] > tree[1][1]]
    data = data[data[tree[3][0]] <= tree[3][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 9
prev = [0,1,4] 
curr = 9
# Update dataframe
try:
    data = data[(data[root] > dic2.get(root))]
    data = data[(data[tree[1][0]] <= tree[1][1])]
    data = data[(data[tree[4][0]] > tree[4][1])]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 10
prev = [0,1,4] 
curr = 10
try:
    data = data[data[root] > dic2.get(root)]
    data = data[data[tree[1][0]] <= tree[1][1]]
    data = data[data[tree[4][0]] <= tree[4][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 11
prev = [0,2,5]
curr = 11
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] > tree[2][1]]
    data = data[data[tree[5][0]] > tree[5][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 12
prev = [0,2,5]
curr = 12
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] > tree[2][1]]
    data = data[data[tree[5][0]] <= tree[5][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 13
prev = [0,2,6]
curr = 13
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] <= tree[2][1]]
    data = data[data[tree[6][0]] > tree[6][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

# Node 14
prev = [0,2,6]
curr = 14
try:
    data = data[data[root] <= dic2.get(root)]
    data = data[data[tree[2][0]] <= tree[2][1]]
    data = data[data[tree[6][0]] <= tree[6][1]]
except:
    pass
# Run node function
node(prev,curr)
data = train

##############################
###### Step 4. Results #######
##############################

print("Classification Tree is ready. Please use Results script to observe accuracy.")