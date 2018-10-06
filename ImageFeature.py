import numpy as np
import ImageModel as imMod
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from create_dataframe import create_df
from collections import defaultdict
import cv2

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

# Dataframe
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

    return fillRatio, formFactor, area

def getGridOfImage():
    image_dict = defaultdict(list)
    fillRatioL = []
    formFactorL = []
    
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
        hue = np.ndarray.flatten(hue).tolist()
        sat = np.ndarray.flatten(sat).tolist()
        val = np.ndarray.flatten(val).tolist()

#        (iImg, jImg, kImg) = np.shape(testImg)
#        i = iImg
#        for line in reversed(testImg):
#            j = jImg
#            for pixel in reversed(line):
#                if all(x > 245 for x in pixel) or all(x < 10 for x in pixel):             
#                    hue.pop(i*j -1)
#                    sat.pop(i*j -1)
#                    val.pop(i*j -1)
#                j -= 1
#            i -= 1

        hueL.extend(hue)
        satL.extend(sat)
        valL.extend(val)    
        

  
    return hueL, satL, valL


if __name__ == '__main__':
    imgType = 'C'
    try:
        testMasks(image_dict[imgType][0])    
    except NameError:
        (image_dict, df) = getGridOfImage()
        testMasks(image_dict[imgType][0])

    df.to_csv('dataset.csv', encoding='utf-8', index=False)
    hue_type, sat_type, val_type = compute_histogram_type("B")
    # Show histograms
    plt.figure(1)
    plt.figure(figsize=(10,8))
    plt.subplot(311)                             
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue")
    plt.hist(hue_type, bins='auto',normed=1)
    plt.subplot(312)                             
    plt.title("Saturation")
    plt.hist(sat_type, bins='auto',normed=1)
    plt.subplot(313)
    plt.title("Luminosity Value")
    plt.hist(val_type, bins='auto',normed=1)
    plt.show()

    plt.figure(2)

    plt.hist2d(hue_type, sat_type, bins=10, range=[[0,255],[0,255]],normed=1, norm=LogNorm())



