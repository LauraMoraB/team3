# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:08 2018

@author: Aitor Sanchez
"""
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 
import pandas as pd
import numpy as np

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
        fillRatioStats[signalType] = compute_freq(signalType, fillRatioList, 'fillRatio', plot, 'green')
        formFactorStats[signalType] = compute_freq(signalType, formFactorList, 'formFactor', plot, 'red')
        areaStats[signalType] = compute_freq(signalType, areaList, 'area', plot, 'black')

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
        reference = sort_by_mean(typeDf.index.values.tolist(), typeDf.Area.tolist())
        k = 0
        for indexRef in reference:
            if(k == 2 or k == 5 or k == 8):
<<<<<<< HEAD
                # validationset
                validation = validation.append(typeDf[typeDf.index.values == name])
            else:
                # trainset
                train = train.append(typeDf[typeDf.index.values == name])
=======
                # validationnset
                validation = validation.append(typeDf[typeDf.index == indexRef])
            else:
                # setset
                train = train.append(typeDf[typeDf.index == indexRef])
>>>>>>> 7e2e46795c93660bc9f293406f58a26ae31ae747
            if(k == 9):
                k = 0
            else:
                k += 1
    return train, validation

if __name__ == '__main__':
    plot = False
    try:
        (train, validation) = split_by_type(df)
    except NameError:
        (image_dict, df) = getGridOfImage()
        (train, validation) = split_by_type(df)

    
    
        
        
        
        
    