# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:52:08 2018

@author: Aitor Sanchez
"""
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 


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
        


if __name__ == '__main__':
    imgType = 'C'        
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

