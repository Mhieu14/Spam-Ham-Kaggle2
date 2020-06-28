
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import Pineline
import os
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

def getInput(url = 'input/emails_dataset.csv', number = 5000):
    dataset = pd.read_csv(url)
    if number != 5000:
        dataset = dataset.sample(number) 
    #Checking for duplicates and removing them
    #dataset.drop_duplicates(inplace = True)
    y = dataset["spam"].to_numpy()
    X = dataset["text"].to_numpy()
    
    return X,y

