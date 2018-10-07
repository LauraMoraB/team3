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
        fillRatioList = []
        formFactorList = []       
        areaList = []       
        for signalGrid in image_dict[signalType]:
            fillRatioList.append(signalGrid.fillRatio)
            formFactorList.append(signalGrid.formFactor)                    
            areaList.append(signalGrid.area)                    
        compute_freq(signalType, fillRatioList, 'fillRatio', plot, 'green')
        compute_freq(signalType, formFactorList, 'formFactor', plot, 'red')
        compute_freq(signalType, areaList, 'area', plot, 'black')
        fillRatioStats[signalType] = fillRatioList
        formFactorStats[signalType] = formFactorList
        areaStats[signalType] = areaList
    return (fillRatioStats, formFactorStats, areaStats)

def compute_freq(signalType, imgInfo, name, plot, color):        

    if(plot == True):
        plt.hist(imgInfo, bins=30, color=color)
        plt.ylabel('f')
        plt.xlabel(name)
        plt.title('signalType '+signalType)
        plt.show()
    
    return (np.mean(imgInfo), np.std(imgInfo))

def sort_by_mean(reference, data):
    dataError = []
    meanData = np.mean(data)

    for value in data:
        dataError.append(abs(value - meanData))
    sortedReference = [x for _,x in sorted(zip(dataError, reference))]

    return sortedReference
        

def split_by_type(dataset):
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type', "Image", "Mask", "FillRatio", "FormFactor", "Area"]
    train = pd.DataFrame(columns=col)
    validation = pd.DataFrame(columns=col)
    for typeSignal in dataset.Type.unique():
        typeDf = dataset[dataset.Type == typeSignal]
        reference = sort_by_mean(typeDf.Image.tolist(), typeDf.Area.tolist())
        k = 0
        print(len(reference))
        for name in reference:
            if(k == 2 or k == 5 or k == 8):
                # trainset
                train = train.append(typeDf[typeDf.Image == name])
            else:
                # validationset
                validation = validation.append(typeDf[typeDf.Image == name])
            if(k == 9):
                k = 0
            else:
                k += 1
    return train, validation

if __name__ == '__main__':
    plot = False
    try:
#        (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot)   
        (train, validation) = split_by_type(df)
    except NameError:
        (image_dict, df) = getGridOfImage()
        (train, validation) = split_by_type(df)

#        (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot)
    
    
        
        
        
        
    