import glob
import numpy as np
import ImageModel as imMod
import matplotlib.pyplot as plt
from create_dataframe import create_df
from collections import defaultdict
import cv2

def getPartialName(txtname):
    pathList =txtname.split(".")
    maskName = pathList[0] +"."+ pathList[1]
    return maskName

def getGridOfMask(df, imageName, i, addPath, addPathMask, addPathGt):
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

    return fillRatio, formFactor, area

def getGridOfImage(df, addPath, addPathMask, addPathGt):
    image_dict = defaultdict(list)
    fillRatioL = []
    formFactorL = []
    areaL = []
    
    for i in range(len(df)): 
        imageName=df["Image"].iloc[i]
        imageTrain = cv2.imread(addPath+imageName,1)
        areaImg = imageTrain[int(df["UpLeft(Y)"].iloc[i]):int(df["DownRight(Y)"].iloc[i]), int(df["UpLeft(X)"].iloc[i]):int(df["DownRight(X)"].iloc[i])]
        
        fillRatio, formFactor, areaMask = getGridOfMask(df, imageName, i, addPath, addPathMask, addPathGt)
        
        areaFinal = cv2.bitwise_and(areaImg,areaImg,mask = areaMask) # Imagen final con la señal solo
        partialName = getPartialName(imageName)
        
        typeSignal = df["Type"].iloc[i]
        (xImg, yImg, zImg) = np.shape(areaFinal)
        area = xImg*yImg
        
        bean = imMod.ModelImage(areaImg, typeSignal, fillRatio, formFactor, partialName, areaMask, areaFinal, area, imageTrain)       
        image_dict[typeSignal].append(bean)
                
        areaL.append(area)
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
       
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
    df["Area"]=areaL
        
    return image_dict, df

def testMasks(img):
    testImg = img.imageGrid
    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
    plt.show()
    finalImg = img.finalGrid
    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
<<<<<<< HEAD
    plt.show()


if __name__ == '__main__':
    imgType = 'C'
    try:
        testMasks(image_dict[imgType][0])    
    except NameError:
        image_dict = getGridOfImage()
        testMasks(image_dict[imgType][0])



=======
    plt.show()
>>>>>>> master
