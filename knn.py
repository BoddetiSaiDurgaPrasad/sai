# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 23:46:19 2023

@author: Sai Durga Prasad
"""

import warnings
warnings.filterwarnings("ignore", message="The default value of `keepdims`")

import pandas as pd

import sklearn

import scipy

import numpy as np

import matplotlib as mp

from scipy import stats

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split as tts

from sklearn.neighbors import KNeighborsClassifier

d=pd.read_csv(r"D:\ml\iris.csv")

d.dropna(inplace=True) 

x=d.iloc[:,:-1]

y=d.iloc[:,-1]
'''
#mode, _ = stats.mode(y[neigh_ind, k], axis=1)'''

xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.20)

model=KNeighborsClassifier(n_neighbors=1)

model.fit(xtrain,ytrain) # to train the machine

ypred=model.predict(xtest)

print(accuracy_score(ytest, ypred))