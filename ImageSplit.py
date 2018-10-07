# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:08 2018

@author: Aitor Sanchez
"""
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 
import pandas as pd
import numpy as np
import cv2
from ImageFeature import getPartialName
from collections import defaultdict



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
        

def split_by_type(dataset, pathimages, pathmask):
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type', "Image", "Mask", "FillRatio", "FormFactor", "Area"]
    train = pd.DataFrame(columns=col)
    validation = pd.DataFrame(columns=col)
    for typeSignal in dataset.Type.unique():
        typeDf = dataset[dataset.Type == typeSignal]
        reference = sort_by_mean(typeDf.index.values.tolist(), typeDf.Area.tolist())
        k = 0
        for indexRef in reference:
            if(k == 2 or k == 5 or k == 8):
                # validationnset
                validation = validation.append(typeDf[typeDf.index == indexRef])
            else:
                # setset
                train = train.append(typeDf[typeDf.index == indexRef])
            if(k == 9):
                k = 0
            else:
                k += 1
                
    # save validation images
    for image in validation["Image"].tolist():  
        imageTrain = cv2.imread(pathimages+image,1)
        cv2.imwrite("./datasets/validation/"+image, imageTrain)
        
    for mask in validation["Mask"].tolist():
        maskTrain = cv2.imread(pathmask+mask,1)
        cv2.imwrite("./datasets/validation/mask/"+mask, maskTrain)
        
    return train, validation


def divide_dictionary(image_dict, dataFrame1, dataFrame2):
    dict1 = defaultdict(list)
    dict2 = defaultdict(list)
    
    for typeSignal in image_dict:
        type1 = dataFrame1[dataFrame1.Type == typeSignal]
        typeName1 = type1.Image.values.tolist()
        typeArea1 = type1.Area.values.tolist()
        type2 = dataFrame2[dataFrame2.Type == typeSignal]
        typeName2 = type2.Image.values.tolist()
        typeArea2 = type2.Area.values.tolist()
        for signal in image_dict[typeSignal]:
            if signal.name+'.jpg' in typeName1 and signal.area in typeArea1:
                dict1[typeSignal].append(signal)
            elif signal.name+'.jpg' in typeName2 and signal.area in typeArea2:
                dict2[typeSignal].append(signal)
                
    return (dict1, dict2)


#if __name__ == '__main__':
#    plot = False
#    try:
#        (train, validation) = split_by_type(df)
#    except NameError:
#        (image_dict, df) = getGridOfImage()
#        (train, validation) = split_by_type(df)

    
    
        
        
        
        
    