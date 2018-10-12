from ImageFeature import get_full_image
import numpy as np
import cv2

def color_segmentation(df, path):
    kernel = np.ones((6,6),np.uint8)
    for i in range(len(df)):       
        # Gets images one by one
        fullImage = get_full_image(df.iloc[i], path)    
        imageName = df['Image'].iloc[i]     
        # Prepares mask files
        sizeImg  = np.shape(fullImage)     
        fullMask = np.zeros((sizeImg[0], sizeImg[1]))
        # Color space change
        fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV)

        hsv_rang= (
             np.array([0,50,60]), np.array([20, 255, 255]) #RED
             ,np.array([300,75,60]), np.array([350, 255, 255]) #DARK RED
             ,np.array([100,50,40]), np.array([140, 255, 255]) #BLUE
        )
        
        size_hsv_rang = np.size(hsv_rang,0)
        for i in range(0, size_hsv_rang-1, 2):
            lower = hsv_rang[i]
            upper = hsv_rang[i+1] 
            for j in range (0,1):
                mask = cv2.inRange(fullImage, lower, upper)
                if (j == 0):
                    maskMorph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                else:
                    maskMorph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                bitwiseRes = cv2.bitwise_and(fullImage, fullImage, mask = maskMorph)
                blur = cv2.blur(bitwiseRes, (5, 5), 0)            
                imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                
                ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
                heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    for cnt in contours:
                        x,y,w,h = cv2.boundingRect(cnt)
                        get_inside_grid_segmentation(x, y, w, h, fullImage, fullMask)

        cv2.imwrite("./ResultMask/"+imageName[:-3]+'png', fullMask)
       
    
def get_inside_grid_segmentation(x, y ,w, h, image, fullMask):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if w > 20 and h > 20 and aspect>0.75:
        imageSegmented = image[y:y+h,x:x+w]
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
            fullMask[y:y+h,x:x+w] = thresh
            ret1, thresh1 = cv2.threshold(greyRes, 0, 255, cv2.THRESH_BINARY)
            image[y:y+h,x:x+w, 0]  =  thresh1
            image[y:y+h,x:x+w, 1]  =  thresh1
            image[y:y+h,x:x+w, 2]  =  thresh1