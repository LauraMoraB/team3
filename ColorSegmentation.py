import glob
import numpy as np
import ImageModel as imMod
import matplotlib.pyplot as plt
from create_dataframe import create_df
from collections import defaultdict
import cv2
from ImageFeature import getPartialName

def colorSegmentation(image_dict):
    imgTypes = ('A','B','C','D','E','F')
    kernel = np.ones((6,6),np.uint8)

    for imgType in imgTypes:
        numberOfItems = np.shape(image_dict[imgType])
        for imageNumber in range(0, numberOfItems[0]-1):

            img = image_dict[imgType][imageNumber]
    
            croped = img.completeImg
            
            sizeFinalImg  = np.shape(croped)
            
            finalMask = np.zeros((sizeFinalImg[0], sizeFinalImg[1]))
            
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
                            getInsideGridSegmentation(x,y,w,h,croped, finalMask)

            cv2.imwrite("./ResultMask_2/"+img.name+'.png', finalMask)
            #cv2.imwrite("./Resultados/"+img.name+'.jpg', croped)
            #plt.imsave("./ResultMask/"+img.name+'.jpg', finalMask)
            #plt.imsave("./Resultados/"+img.name+'.jpg', croped)
            
def colorSegmentation_test(df_test, addPath):
    kernel = np.ones((6,6),np.uint8)
    for image in df_test["Image"].tolist():       
        
        
        imageTrain = cv2.imread(addPath+image,1)
        image = getPartialName(image)
        croped = imageTrain
        
        sizeFinalImg  = np.shape(croped)
        
        finalMask = np.zeros((sizeFinalImg[0], sizeFinalImg[1]))
        
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
                        getInsideGridSegmentation(x,y,w,h,croped, finalMask)

        cv2.imwrite("./ResultMask/"+image+'.png', finalMask)
       
    
def getInsideGridSegmentation(x,y,w,h, cropedSegment, finalMask):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if w > 20 and h > 20 and aspect>0.75:
        imageSegmented = cropedSegment[y:y+h,x:x+w]
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
            if i==0:
                maskConcatenated = mask
            else:
                maskConcatenated = cv2.add(maskConcatenated, mask)
        
        bitwiseRes = cv2.bitwise_and(testCropHSVSegmented, testCropHSVSegmented, mask = maskConcatenated)
    
        greyRes  = cv2.cvtColor(bitwiseRes, cv2.COLOR_BGR2GRAY)
        fillRatioOnes = np.count_nonzero(greyRes)
        sizeMatrix = np.shape(greyRes)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        if fillRatio > 0.5:
            ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
            finalMask[y:y+h,x:x+w] = thresh
            ret1, thresh1 = cv2.threshold(greyRes, 0, 255, cv2.THRESH_BINARY)
            cropedSegment[y:y+h,x:x+w, 0]  =  thresh1
            cropedSegment[y:y+h,x:x+w, 1]  =  thresh1
            cropedSegment[y:y+h,x:x+w, 2]  =  thresh1