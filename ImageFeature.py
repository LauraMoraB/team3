__author__ = 'Zaius'

import cv2
import glob
import numpy as np
import os
import ImageModel as imMod
from collections import defaultdict
from matplotlib import pyplot as plt

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

mask_location_list = []
mask_list = []
        
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


def load_annotations(annot_file):
    # Annotations are stored in text files containing
    # the coordinates of the corners (top-left and bottom-right) of
    # the bounding box plus an alfanumeric code indicating the signal type:
    # tly, tlx, bry,brx, code
    annotations = []

    for line in open(annot_file).read().splitlines():

        annot_values = line.split()
        annot_values = [x.strip() for x in annot_values]
        for ii in range(4):
            annot_values[ii] = float(annot_values[ii])
        annotations.append(annot_values)
        
    return annotations



def getGridOfMask(imageName, values):
          
    maskName = getMaskFileName(imageName)
    mask = cv2.imread(maskName,0)
    area = mask[int(float(values[0])):int(float(values[2])), int(float(values[1])):int(float(values[3]))]
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
    for imageName in glob.glob(addPath+'*.jpg'):      
        txtname = getGtFileName(imageName)
        content = load_annotations(txtname)
        for values in content: 
            imageTrain = cv2.imread(imageName,1)
            areaImg = imageTrain[int(float(values[0])):int(float(values[2])), int(float(values[1])):int(float(values[3]))]
            fillRatio, formFactor, areaMask = getGridOfMask(imageName, values)
            areaFinal = cv2.bitwise_and(areaImg,areaImg, mask = areaMask)           
            #areaFinal = areaImg * areaMask
            
            partialName = getPartialName(imageName)
            typeSignal = values[4].rstrip()
            bean = imMod.ModelImage(areaImg, typeSignal, fillRatio, formFactor, partialName, areaMask, areaFinal)       
            image_dict[typeSignal].append(bean)
    
    return image_dict

def testMasks(img):
    testImg = img.imageGrid
    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
    plt.show()
    finalImg = img.finalGrid
    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    imgType = 'B'
    try:
        testMasks(image_dict[imgType][0])    
    except NameError:
        image_dict = getGridOfImage()
        testMasks(image_dict[imgType][0])




<<<<<<< HEAD
=======
if __name__ == '__main__':
    imgType = 'C'
    try:
        testMasks(image_dict[imgType][0])    
    except NameError:
        image_dict = getGridOfImage()
        testMasks(image_dict[imgType][0])

    image_dict, df = getGridOfImage()
    df.to_csv('dataset.csv', encoding='utf-8', index=False)
    hueTotal=[]
    satTotal=[]
    valTotal=[]
    for file in df.Type.unique():
        hue_type, sat_type, val_type = compute_histogram_type(file)
        hueTotal.append(hue_type)
        satTotal.append(sat_type)
        valTotal.append(val_type)

    # Show histograms
    plt.figure(figsize=(10,8))
    plt.subplot(311)                             
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue")
    plt.hist(hueTotal, bins=180)
    plt.subplot(312)                             
    plt.title("Saturation")
    plt.hist(satTotal, bins=128)
    plt.subplot(313)
    plt.title("Luminosity Value")
    plt.hist(valTotal, bins=128)
    plt.show()

>>>>>>> laura-dev
