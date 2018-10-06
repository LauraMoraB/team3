
# coding: utf-8

# In[90]:


# %load ImageSplit.py
"""
Created on Thu Oct  4 19:52:08 2018

@author: Aitor Sanchez
"""

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 
from create_dataframe import create_df
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset.csv')

def computeStats(image_dict):
    #Stadistical study for the different signal types in order to properly
    #split the training set into two sets,  ~70% and ~30% with the best 
    #main features represented in both of them
    fillRatio_dict = {}
    formFactor_dict = {}
    
    for signalType in image_dict:
        fillRatio_list = []
        formFactor_list = []
        
        for signalGrid in image_dict[signalType]:
            fillRatio_list.append(signalGrid.fillRatio)
            formFactor_list.append(signalGrid.formFactor)
            
        fillRatio_dict[signalType] = fillRatio_list
        formFactor_dict[signalType] = formFactor_list
    
    return (fillRatio_dict, formFactor_dict)
        

def compute_freq(imgType):        
    try:
        (fillRatio_dict, formFactor_dict) = computeStats(image_dict)   
    except NameError:
        image_dict = getGridOfImage()
        (fillRatio_dict, formFactor_dict) = computeStats(image_dict)
    
    plt.hist(fillRatio_dict[imgType])
    plt.ylabel('frequencia')
    plt.xlabel('fillRatio')
    plt.title('signalType '+imgType)
    plt.show()

def split_by_type():
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type', "Image", "Mask", "FillRatio", "FormFactor"]
    train = pd.DataFrame(columns=col)
    validation = pd.DataFrame(columns=col)
    for type in df.Type.unique():
        typeDf = df[df.Type == type]
        train1, validation1 = train_test_split(typeDf, test_size=0.3)
       
        train = pd.concat([train, train1],ignore_index=True)
        validation = pd.concat([validation, validation1],ignore_index=True)        
    
    return train, validation

if __name__ == '__main__':
    train,validation = split_by_type()

