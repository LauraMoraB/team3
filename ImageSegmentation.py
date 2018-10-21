from ImageFeature import get_full_image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def apply_morphology_operations(mask):

    kernelOpen = np.ones((5,5),np.uint8)
    kernelClose = np.ones((5,5),np.uint8)
    kernelErode = np.ones((10,10),np.uint8)
    kernelDialte  = np.ones((10,10),np.uint8)
            
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)
    
    mask = cv2.dilate(mask,kernelDialte, iterations = 1)
    mask = cv2.erode(mask ,kernelErode, iterations = 1)
    
    return mask
    

def apply_color_mask(fullImage, hsv_rang):
    
    size_hsv_rang = np.size(hsv_rang,0)
    for i in range(0, size_hsv_rang-1, 2):
        lower = hsv_rang[i]
        upper = hsv_rang[i+1] 
        
        mask = cv2.inRange(fullImage, lower, upper)
        if i==0:
            maskConcatenated = mask
        else:
            maskConcatenated = cv2.add(maskConcatenated, mask)
            
    return maskConcatenated
    



def color_segmentation(df, path, hsv_rang):

    listOfBB = []
    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        imgBGR = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']   

        # Prepares mask files
        sizeImg  = np.shape(imgBGR)     
        fullMask = np.zeros((sizeImg[0], sizeImg[1]))
        # Color space change, it operates on HSV, RGB 
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        imgRGB = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)

        # Get mask, color + morphology         
        color_mask = apply_color_mask(imgHSV, hsv_rang)
        morphology_mask = apply_morphology_operations(color_mask)         
#        bitwiseRes = cv2.bitwise_and(imgHSV, imgHSV, mask = morphology_mask)        
# 
#        blur = cv2.blur(bitwiseRes, (5, 5), 0)            
#        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
#        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
#        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        if contours:
#            for cnt in contours:
#                x,y,w,h = cv2.boundingRect(cnt)
#                cv2.drawContours(bitwiseRes, [cnt], 0,255,-1)
#                get_inside_grid_segmentation(x, y, w, h, imgRGB, fullMask, bitwiseRes, listOfBB, imageName[:-4])
#        
        create_maskFolders('train')
        create_maskFolders('validation') 
        cv2.imwrite(path+'resultMask/colorMask/mask.'+imageName[:-3]+'png', color_mask)
        cv2.imwrite(path+'resultMask/morphologyMask/mask.'+imageName[:-3]+'png', morphology_mask)
#        cv2.imwrite(path+'resultMask/finalMask/mask.'+imageName[:-3]+'png', fullMask)
#        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
#        cv2.imwrite(path+'resultMask/resBB.'+imageName[:-3]+'png', imgBGR)
    
    return listOfBB


def get_inside_grid_segmentation(x, y ,w, h, image, fullMask, currentMask, listOfBB, imageName):
    window = []
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
        
    if (280 > w > 30) and (280  >h > 30) and aspect>0.75:
        
        bitwiseResSegmented = currentMask[y:y+h,x:x+w]
    
        greyRes  = cv2.cvtColor(bitwiseResSegmented, cv2.COLOR_BGR2GRAY)
        fillRatioOnes = np.count_nonzero(greyRes)
        sizeMatrix = np.shape(greyRes)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        
        if fillRatio > 0.45:
            
            window.append([y,x,h+y,w+x])
            
            listOfBB.append((imageName,window))

            ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
            
            fullMask[y:y+h,x:x+w] = thresh
            
            get_templete_matching(x, y ,w, h, thresh, image)

            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),10)

                
def get_region_of_interest(x, y ,w, h, currentMask):
    
    totalY, totalX = np.shape(currentMask[:,:,0])
    aditionalWidth = int(0.1 * w)
    aditionalHeight = int(0.1 * h)
    
    y1 = y
    x1 = x
    y2 = y + h
    x2 = x + w
    
    if (x1 - aditionalWidth) > 0:
        x1 = x1 - aditionalWidth
    if (x2 + aditionalWidth) < totalX:
        x2 = x2 + aditionalWidth
    if (y1 - aditionalHeight) > 0:
        y1 = y1 + aditionalHeight
    if (y2 + aditionalHeight) < totalY:
        y2 = y2 + aditionalHeight
        
    return currentMask[y1:y2,x1:x2]
            
def get_templete_matching(x, y ,w, h, maskSegmented, image):
    for i in range(1,5):
        maskTemplate = cv2.imread("template/mask.temp"+str(i)+".png",0)
        maskTemplate = cv2.resize(maskTemplate,(w-1,h-1))
        # Perform match operations. 
        res = cv2.matchTemplate(maskSegmented, maskTemplate, cv2.TM_CCOEFF_NORMED) 
          
        # Specify a threshold 
        threshold = 0.8
          
        # Store the coordinates of matched area in a numpy array 
        loc = np.where( res >= threshold)  
          
        # Draw a rectangle around the matched region. 
        for pt in zip(*loc[::-1]): 
#            cv2.rectangle(image, (pt[0]+x, pt[1]+y), (pt[0] + x + w, pt[1] + y + h), (0,255,0), 5)
            if(i==1):
                cv2.putText(image, 'CIRCULO', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
            elif(i==2):
                cv2.putText(image, 'RECTANGLE', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
            elif(i==3):
                cv2.putText(image, 'TRIANGLE', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)
            else:
                cv2.putText(image, 'WARNING', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 3)


def create_maskFolders(split_name):   
    path = "datasets/"
    pathStructure = [['split/'],[split_name+'/'],['resultMask/'],['colorMask/','morphologyMask/','finalMask/']]
    for paths in pathStructure:
        for subPath in paths:
            if not os.path.exists(path+str(subPath)):
                os.makedirs(path+subPath)
        path = path+subPath  
        
        