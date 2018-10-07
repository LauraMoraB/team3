
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
            
            partialName = getPartialName(imageName)
            typeSignal = values[4].rstrip()
            bean = imMod.ModelImage(areaImg, typeSignal, fillRatio, formFactor, partialName, areaMask, areaFinal, imageTrain)       
            image_dict[typeSignal].append(bean)
    
    return image_dict

def testMasks(img):
    testImg = img.imageGrid
    plt.imshow(cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB))
    plt.show()
    finalImg = img.finalGrid
    plt.imshow(cv2.cvtColor(finalImg, cv2.COLOR_BGR2RGB))
    plt.show()
    completeImg = img.completeImg
    plt.imshow(cv2.cvtColor(completeImg, cv2.COLOR_BGR2RGB))
    plt.show()



def colorSegmentation(image_dict):
#    imgTypes = ('A','B','C','D','E','F')
    imgTypes = ('D')
    kernel = np.ones((6,6),np.uint8)
    for imgType in imgTypes:
        numberOfItems = np.shape(image_dict[imgType])
        for imageNumber in range(0, numberOfItems[0]-1):
#        for imageNumber in range(0, 1):

            img = image_dict[imgType][imageNumber]
    
            croped = img.finalGrid
            testCropHSV = cv2.cvtColor(croped, cv2.COLOR_BGR2HSV)
    
            hsv_rang= (
                 np.array([0,50,60]), np.array([20, 255, 255]) #RED
                 ,np.array([300,75,60]), np.array([350, 255, 255]) #DARK RED
                 ,np.array([100,50,40]), np.array([140, 255, 255]) #BLUE

            )
            size_hsv_rang = np.size(hsv_rang,0)
            for i in range(0, size_hsv_rang-1,2):
                lower = hsv_rang[i]
                upper = hsv_rang[i+1] 
                for j in range (0,1):
                    mask = cv2.inRange(testCropHSV, lower, upper)
                    if (j == 0):
                        maskMorph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    else:
                        maskMorph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    bitwiseRes = cv2.bitwise_and(testCropHSV, testCropHSV, mask = maskMorph)
                    blur = cv2.blur(bitwiseRes, (5, 5), 0)            
                    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                    
                    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
                    heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        for cnt in contours:
                            x,y,w,h = cv2.boundingRect(cnt)
                            getInsideGridSegmentation(x,y,w,h,croped)
#                            rectangleImg = cv2.rectangle(croped,(x,y),(x+w,y+h),(0,255,0),2)
#                        plt.imshow(rectangleImg)
#                        plt.show()
#                    croped = cv2.drawContours(croped, contours, -1 ,(0,255,0), 3)
#            plt.imshow(croped)
#            plt.show()
    
def getInsideGridSegmentation(x,y,w,h, croped):
    imageSegmented = croped[y:y+h,x:x+w]
    testCropHSVSegmented = cv2.cvtColor(imageSegmented, cv2.COLOR_BGR2HSV)
    hsv_rang_Seg= (
         np.array([0,50,60]), np.array([20, 255, 255]) #RED
         ,np.array([300,75,60]), np.array([350, 255, 255]) #DARK RED
         ,np.array([100,50,40]), np.array([140, 255, 255]) #BLUE
         ,np.array([0,0,0]), np.array([180, 255, 30]) #BLACK
         ,np.array([0,0,200]), np.array([180, 255, 255]) #WHITE
    )
 
    ize_hsv_rang_seg = np.size(hsv_rang_Seg ,0)
    for i in range(0, ize_hsv_rang_seg-1,2):
        lower = hsv_rang_Seg[i]
        upper = hsv_rang_Seg[i+1] 
        mask = cv2.inRange(testCropHSVSegmented, lower, upper)
#        if not maskConcatenated:
#            maskConcatenated = mask
#        else:
#            maskConcatenated = cv2.add(maskConcatenated, mask)

    bitwiseRes = cv2.bitwise_and(testCropHSVSegmented, testCropHSVSegmented, mask = mask)
    cv2.imshow(bitwiseRes)
    cv2.show()
#    fillRatioOnes = np.count_nonzero(result)
#    sizeMatrix = np.shape(result)
#    fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
#    fillRatio = fillRatioOnes/fillRatioZeros
#    print(fillRatio)
    
def getHistogram(image_dict):
    imgTypes = ('A','B','C','D','E','F')
    for imgType in imgTypes:
        numberOfItems = np.shape(image_dict[imgType])
        for imageNumber in range(0, numberOfItems[0]-1):
            img = image_dict[imgType][imageNumber]
            testImg = img.finalGrid
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([testImg],[i],None,[256],[1,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()

if __name__ == '__main__':
    imgTypes = ('A','B','C','D','E','F')
    imgType = imgTypes[0]
    try:

#        getHistogram(image_dict)
        colorSegmentation(image_dict) 
#        testMasks(image_dict[imgType][0])    
    except NameError:
        image_dict = getGridOfImage()
        testMasks(image_dict[imgType][0])




