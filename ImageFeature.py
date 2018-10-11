import numpy as np
import cv2

def getFullImage(path, dfSingle):
    return cv2.imread(path + dfSingle['Image'],1)

def getCroppedImage(path, dfSingle):
    image = getFullImage(path, dfSingle)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def getFullMask(path, dfSingle):
    return cv2.imread(path+'mask/' + dfSingle['Mask'], 0)

def getCroppedMask(path, dfSingle):
    image = getFullMask(path, dfSingle)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def getMaskAspect(path, dfSingle): 
    crop = getCroppedMask(path, dfSingle)  
    (imgX, imgY) = np.shape(crop)
    imgOnes = np.count_nonzero(crop)    
    imgArea = imgX*imgY
    imgFillRatio = imgOnes/imgArea
    if(imgX < imgY):    
        imgFormFactor = abs(imgX/imgY)
    else:
        imgFormFactor = abs(imgY/imgX)

    return imgFillRatio, imgFormFactor, imgArea

def getGridOfImage(df, path):
    fillRatioL = []
    formFactorL = []
    areaL = []
    
    for i in range(len(df)):       
        fillRatio, formFactor, area = getMaskAspect(path, df.iloc[i])    
#        areaFinal = cv2.bitwise_and(areaImg,areaImg,mask = areaMask) # Imagen final con la señal solo                   
        areaL.append(area)
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
       
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
    df["Area"]=areaL
        
    return df
