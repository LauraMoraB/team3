import numpy as np
import ImageModel as imMod
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2

def getPartialName(txtname):
    #Returns Partian name for a given image data file of the dataset
    # e.g. for forest.1034.png returs forest.1034
    pathList = txtname.split(".")
    maskName = pathList[0] +"."+ pathList[1]
    return maskName

def getFullImage(path, dfSingle):
    return cv2.imread(path + dfSingle['Image'],1)

def getCroppedImage(path, dfSingle):
    image = getFullImage(path, dfSingle)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def getFullMask(path, dfSingle):
    return cv2.imread(path+'mask' + dfSingle['Mask'],1)

def getCroppedMask(path, dfSingle):
    image = getFullMask(path, dfSingle)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def getMaskAspect(path, dfSingle):
    crop = getCroppedMask(path+'mask', dfSingle)  

    (imgX, imgY, imgZ) = np.shape(crop)
    imgOnes = np.count_nonzero(crop)    
    imgArea = imgX*imgY
    imgFillRatio = imgOnes/imgArea
    if(imgX < imgY):    
        imgFormFactor = abs(imgX/imgY)
    else:
        imgFormFactor = abs(imgY/imgX)

    return imgFillRatio, imgFormFactor, imgArea

def getGridOfImage(df, addPath, addPathMask, addPathGt):
    fillRatioL = []
    formFactorL = []
    areaL = []
    
    for i in range(len(df)):       
        fillRatio, formFactor, area = getMaskAspect(df.iloc[i], addPathMask)        
#        areaFinal = cv2.bitwise_and(areaImg,areaImg,mask = areaMask) # Imagen final con la señal solo                   
        areaL.append(area)
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
       
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
    df["Area"]=areaL
        
    return df

def testMasks(img):
    testImg = img.imageGrid
    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
    plt.show()
    finalImg = img.finalGrid
    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
    plt.show()