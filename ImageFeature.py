
# coding: utf-8

# In[58]:


# %load ImageFeature.py
import cv2
import glob
import numpy as np
import os
import ImageModel as imMod
from matplotlib import pyplot as plt
from create_dataframe import create_df
from collections import defaultdict

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

# Create Dataframe
df = create_df(addPath, addPathMask, addPathGt)

def getPartialName(txtname):
    pathList =txtname.split(".")
    maskName = pathList[0] +"."+ pathList[1]
    return maskName

def getGridOfMask(imageName, i):
    maskName=df["Mask"].iloc[i]
    mask = cv2.imread(addPathMask+maskName,0)
    area = mask[int(df["UpLeft(Y)"].iloc[i]):int(df["DownRight(Y)"].iloc[i]), int(df["UpLeft(X)"].iloc[i]):int(df["DownRight(X)"].iloc[i])]
    
    fillRatioOnes = np.count_nonzero(area)
    sizeMatrix = np.shape(area)
    
    fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
    fillRatio = fillRatioOnes/fillRatioZeros
    
    if(sizeMatrix[0]<sizeMatrix[1]):    
        formFactor = abs(sizeMatrix[0]/sizeMatrix[1])
    else:
        formFactor = abs(sizeMatrix[1]/sizeMatrix[0])

    return fillRatio, formFactor, area#, areaSign

def getGridOfImage():
    image_dict = defaultdict(list)
    fillRatioL = []
    formFactorL = []
    areaSignL=[]
    
    for i in range(len(df)): 
        imageName=df["Image"].iloc[i]
        imageTrain = cv2.imread(addPath+imageName,1)
        areaImg = imageTrain[int(df["UpLeft(Y)"].iloc[i]):int(df["DownRight(Y)"].iloc[i]), int(df["UpLeft(X)"].iloc[i]):int(df["DownRight(X)"].iloc[i])]
        
        fillRatio, formFactor, areaMask = getGridOfMask(imageName, i)
        
        areaFinal = cv2.bitwise_and(areaImg,areaImg,mask = areaMask) # Imagen final con la señal solo
        partialName = getPartialName(imageName)
        
        typeSignal = df["Type"].iloc[i]
        bean = imMod.ModelImage(areaImg, typeSignal, fillRatio, formFactor, partialName, areaMask, areaFinal)       
        image_dict[typeSignal].append(bean)
                
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
       
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
        
    return image_dict, df
    

def testMasks(img):
    testImg = img.imageGrid
    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
    plt.show()
    finalImg = img.finalGrid
    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
    plt.show()

def compute_histogram_type(signal_type):
    hueL=[]
    satL=[]
    valL=[]
    for i in range((len(image_dict[signal_type]))):
        img = image_dict[signal_type][i]
        testImg = img.finalGrid

        hsv = cv2.cvtColor(testImg, cv2.COLOR_BGR2HSV)

        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        hueL.append(np.ndarray.flatten(hue))
        satL.append(np.ndarray.flatten(sat))
        valL.append(np.ndarray.flatten(val))
        
    return hueL, satL, valL

if __name__ == '__main__':
    imgType = 'C'
    try:
        testMasks(image_dict[imgType][0])    
    except NameError:
        image_dict = getGridOfImage()
        testMasks(image_dict[imgType][0])

    # save    
    df.to_csv('dataset.csv', encoding='utf-8', index=False)

    # TIPU A
    hue_a, sat_a, val_a = compute_histogram_type("A")
    hue_b, sat_b, val_b = compute_histogram_type("B")
    hue_c, sat_c, val_c = compute_histogram_type("C")
    hue_d, sat_d, val_d = compute_histogram_type("D")
    hue_e, sat_e, val_e = compute_histogram_type("E")
    hue_f, sat_f, val_f = compute_histogram_type("F")

    #Plot histogram
    plt.figure(figsize=(10,8))
    plt.subplot(311)                             
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue A")
    plt.hist(hue_a, bins='auto')
    plt.subplot(312)                             
    plt.title("Saturation A")
    plt.hist(sat_a, bins='auto')
    plt.subplot(313)
    plt.title("Luminosity A")
    plt.hist(val_a, bins='auto')
    plt.show()

