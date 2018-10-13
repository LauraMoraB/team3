# -*- coding: utf-8 -*-
"""
@author: Zaius
"""
import cv2
import numpy as np
pathToMasks = "datasets/train/mask/"
pathToResults = "ResultMask/"

        
def validation(image_dict):    
    imgTypes = ('A','B','C','D','E','F')
    TrueNeg = 0
    TruePos = 0
    FalsePos = 0
    FalseNeg = 0
    print("1")
    for imgType in imgTypes:
        print("2")

        numberOfItems = np.shape(image_dict[imgType])
        for imageNumber in range(0, numberOfItems[0]-1):
            print("3")
            img = image_dict[imgType][imageNumber]
    
            image = img.name
            
            maskResultName = pathToResults +  image + ".png"
            maskResult = cv2.imread(maskResultName,1)
            maskResult = cv2.cvtColor(maskResult, cv2.COLOR_BGR2GRAY)
            ret, maskResult = cv2.threshold(maskResult, 0, 10, cv2.THRESH_BINARY)
            maskResult = maskResult*10
            maskValidatorName = pathToMasks +"mask." +image +".png"
            maskValidator = cv2.imread(maskValidatorName,1)
            maskValidator = cv2.cvtColor(maskValidator, cv2.COLOR_BGR2GRAY)
            ret, maskValidator = cv2.threshold(maskValidator, 0, 10, cv2.THRESH_BINARY)
            maskValidator = maskValidator*10
            dst = cv2.addWeighted(maskValidator,0.7,maskResult,0.3,0)
            unique, counts = np.unique(dst, return_counts=True)
            dict_FScore = dict(zip(unique, counts))

            for value in unique:
                if value == 0:
                    TrueNeg = TrueNeg + dict_FScore[0]
                elif value == 100:
                    TruePos = TruePos + dict_FScore[100]
                elif value == 30:
                    FalsePos = FalsePos + dict_FScore[30]
                elif value == 70:
                    FalseNeg = FalseNeg + dict_FScore[70]
                else:
                    print("NOP")
                
            
    print("True Negative :" + str(TrueNeg))
    print("True Positive :" + str(TruePos))
    print("False Negative :" + str(FalseNeg))
    print("False Positive :" + str(FalsePos))

            