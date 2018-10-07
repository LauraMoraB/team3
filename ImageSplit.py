# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:08 2018

@author: Aitor Sanchez
"""
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def compute_stats(image_dict, plot = False):
    #Stadistical study for the different signal types in order to properly
    #split the training set into two sets,  ~70% and ~30% with the best 
    #main features represented in both of them
    fillRatioStats = {}
    formFactorStats = {}
    areaStats = {}
    for signalType in image_dict:
        fillRatio_list = []
        formFactor_list = []       
        area_list = []       
        for signalGrid in image_dict[signalType]:
            fillRatio_list.append(signalGrid.fillRatio)
            formFactor_list.append(signalGrid.formFactor)                    
            area_list.append(signalGrid.area)                    
        fillRatioStats[signalType] = compute_freq(signalType, fillRatio_list, 'fillRatio', plot, 'green')
        formFactorStats[signalType] = compute_freq(signalType, formFactor_list, 'formFactor', plot, 'red')
        areaStats[signalType] = compute_freq(signalType, area_list, 'area', plot, 'black')
    return (fillRatioStats, formFactorStats, areaStats)
        
def compute_freq(signalType, imgInfo, name, plot, color):        

    if(plot == True):
        plt.hist(imgInfo, bins=30, color=color)
        plt.ylabel('f')
        plt.xlabel(name)
        plt.title('signalType '+signalType)
        plt.show()
    
    return (np.mean(imgInfo), np.std(imgInfo))

def sort_from_mean(reference, data):
    dataError = []
    meanData = np.mean(data)

    for value in data:
        dataError.append(abs(value - meanData))
    sortedReference = [x for _,x in sorted(zip(dataError, reference))]

    return sortedReference
        

def split_by_type(dataset):
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type', "Image", "Mask", "FillRatio", "FormFactor"]
    train = pd.DataFrame(columns=col)
    validation = pd.DataFrame(columns=col)
    for type in dataset.Type.unique():
        typeDf = dataset[dataset.Type == type]
        train1, validation1 = train_test_split(typeDf, test_size=0.3)
       
        train = pd.concat([train, train1],ignore_index=True)
        validation = pd.concat([validation, validation1],ignore_index=True)        
    
    return train, validation

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    train,validation = split_by_type(df)
    plot = False
    try:
        (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot)   
    except NameError:
        (image_dict, df) = getGridOfImage()
        (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot)