# Undergraduate Thesis (MTH40A/B): Lending Club 
# By: Paul Desroches (500699067) at Ryerson University

# Objective: To predict whether a loan will default,  
# using a Decision Tree involving specified features.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Generate list of 3 unique digits ranging from 1 to 9

my_list = list(range(1,10)) # list of integers from 1 to 99
                              # adjust this boundaries to fit your needs
random.shuffle(my_list)
print(my_list[:3]) # <- List of unique random numbers
