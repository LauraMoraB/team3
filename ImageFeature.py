__author__ = 'Zaius'


import cv2
import glob
import numpy as np
import os
import ImageModel as imMod
from matplotlib import pyplot as plt

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

mask_location_list = []
mask_list = []
image_list =  []

signal_A = []
signal_B = []
signal_C = []
signal_D = []
signal_E = []
signal_F = []
        
def getMaskFileName(txtname):
    pathList = txtname.split(os.sep)
    nameFileList = pathList[1].split(".")
    maskName = addPathMask + "mask." + nameFileList[0] +"."+ nameFileList[1] + ".png"
    return maskName

def getGtFileName(txtname):
    pathList = txtname.split(os.sep)
    nameFileList = pathList[1].split(".")
    maskName = addPathGt + "gt." + nameFileList[0] +"."+ nameFileList[1] + ".txt"
    return maskName

def getPartialName(txtname):
    pathList = txtname.split(os.sep)
    nameFileList = pathList[1].split(".")
    maskName = nameFileList[0] +"."+ nameFileList[1]
    return maskName

#def getCompleteMergedImage():
#    for imageName in glob.glob(addPath+ '*.jpg'):
#        rgbImg = cv2.imread(imageName)
#        maskName = getMaskFileName(imageName)
#        maskImg = cv2.imread(maskName)
#        
#        mergedImg = rgbImg * maskImg
#
#        txtname = getGtFileName(imageName)
#        txtfile = open(txtname, "r")
#        content = txtfile.readlines()
#        values = []
#        for x in content:
#            values = x.split(" ")    
#        areaFinalImg = mergedImg[int(float(values[0])):int(float(values[2])), int(float(values[1])):int(float(values[3]))]
#        
#        partialName = getPartialName(imageName)
#
#        plt.imshow(cv2.cvtColor(areaFinalImg, cv2.COLOR_BGR2RGB))
#        plt.suptitle(values[4]+ "_"+ partialName)
#        plt.show()


def getGridOfMask(imageName):
    
    txtname = getGtFileName(imageName)
    txtfile = open(txtname, "r")
    content = txtfile.readlines()
    values = []
    for x in content:
        values = x.split(" ")
        
    maskName = getMaskFileName(imageName)
    mask = cv2.imread(maskName,1)
    area = mask[int(float(values[0])):int(float(values[2])), int(float(values[1])):int(float(values[3]))]
    fillRatioOnes = np.count_nonzero(area[:,:,0])
    sizeMatrix = np.shape(area[:,:,0])
    fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
    fillRatio = fillRatioOnes/fillRatioZeros
    
    if(sizeMatrix[0]<sizeMatrix[1]):    
        formFactor = abs(sizeMatrix[0]/sizeMatrix[1])
    else:
        formFactor = abs(sizeMatrix[1]/sizeMatrix[0])
        
    return fillRatio, formFactor, area
        
def getGridOfImage():
    for imageName in glob.glob(addPath+'*.jpg'):
        
        txtname = getGtFileName(imageName)
        txtfile = open(txtname, "r")
        content = txtfile.readlines()
        values = []
        for x in content:
            values = x.split(" ")
    
        imageTrain = cv2.imread(imageName,1)
        areaImg = imageTrain[int(float(values[0])):int(float(values[2])), int(float(values[1])):int(float(values[3]))]
        fillRatio, formFactor, areaMask = getGridOfMask(imageName)
        
        areaFinal = areaImg * areaMask
        
        partialName = getPartialName(imageName)
        typeSignal = values[4].rstrip()
        bean = imMod.ModelImage(areaImg, typeSignal, fillRatio, formFactor, partialName, areaMask, areaFinal)
        image_list.append(bean)
        if typeSignal  == 'A':
            signal_A.append(bean)
        elif typeSignal == 'B':
            signal_B.append(bean)
        elif typeSignal == 'C':
            signal_C.append(bean)
        elif typeSignal == 'D':
            signal_D.append(bean)
        elif typeSignal == 'E':
            signal_E.append(bean)
        elif typeSignal == 'F':
            signal_F.append(bean)
        else:
            print("NONE of type:"+typeSignal+":")
            
#        plt.imshow(cv2.cvtColor(areaFinal, cv2.COLOR_BGR2RGB))
#        plt.suptitle(values[4])
#        plt.show()
#
#def testMasks():
#    testImg = signal_C[0].imageGrid
#    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
#    plt.show()
#    finalImg = signal_C[0].finalGrid
#    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
#    plt.show()
#    imgMask= signal_C[0].maskGrid
#    plt.imshow(cv2.cvtColor(imgMask, cv2.COLOR_BGR2RGB))
#    plt.show()

getGridOfImage()




