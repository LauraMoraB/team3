from ImageFeature import get_full_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_color_mask(fullImage):
    kernelOpen = np.ones((5,5),np.uint8)
    kernelClose = np.ones((5,5),np.uint8)
    kernelErode = np.ones((3,3),np.uint8)
    kernelDialte  = np.ones((3,3),np.uint8)
    
#    hsv_rang= (
#         np.array([0,50,60]), np.array([20, 255, 255]) #RED
#         ,np.array([300,75,60]), np.array([350, 255, 255]) #DARK RED
#         ,np.array([100,50,40]), np.array([140, 255, 255]) #BLUE
#    )
    hsv_rang= (
         np.array([0,150,50]), np.array([20, 255, 255]) #RED
         ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
         ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
    )
    
    size_hsv_rang = np.size(hsv_rang,0)
    for i in range(0, size_hsv_rang-1, 2):
        lower = hsv_rang[i]
        upper = hsv_rang[i+1] 
        
        mask = cv2.inRange(fullImage, lower, upper)
        if i==0:
            maskConcatenated = mask
        else:
            maskConcatenated = cv2.add(maskConcatenated, mask)
            
    maskConcatenated = cv2.morphologyEx(maskConcatenated, cv2.MORPH_OPEN, kernelOpen)
    maskConcatenated = cv2.morphologyEx(maskConcatenated, cv2.MORPH_CLOSE, kernelClose)
    
    maskConcatenated = cv2.erode(maskConcatenated ,kernelErode, iterations = 1)
    maskConcatenated = cv2.dilate(maskConcatenated,kernelDialte, iterations = 1)

    return cv2.bitwise_and(fullImage, fullImage, mask = maskConcatenated)

def color_segmentation(df, path):


    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        fullImage = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']   
        # Prepares mask files
        sizeImg  = np.shape(fullImage)     
        fullMask = np.zeros((sizeImg[0], sizeImg[1]))
        # Color space change
        plt.imshow(fullImage)
        plt.show()
        fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV)
        copyOfFullImage = fullImage
        
        bitwiseRes = apply_color_mask(fullImage)
#        plt.imshow(bitwiseRes)
#        plt.show()
        
        blur = cv2.blur(bitwiseRes, (5, 5), 0)            
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                get_inside_grid_segmentation(x, y, w, h, copyOfFullImage, fullMask)
                        
        rgbimg = cv2.cvtColor(copyOfFullImage, cv2.COLOR_HSV2RGB)
#        plt.imshow(rgbimg)
#        plt.show()
#        plt.imshow(fullMask)
#        plt.show()
        cv2.imwrite(path+'resultMask/mask.'+imageName[:-3]+'png', fullMask)
        
def apply_mask_color_Inner(imageSegmented):
    hsv_rang_Seg= (
         np.array([0,150,50]), np.array([20, 255, 255]) #RED
         ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
         ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
         ,np.array([0,0,0]), np.array([180, 255, 30]) #BLACK
         ,np.array([0,0,200]), np.array([180, 255, 255]) #WHITE
    )
    ize_hsv_rang_seg = np.size(hsv_rang_Seg ,0)
    for i in range(0, ize_hsv_rang_seg-1,2):
        lower = hsv_rang_Seg[i]
        upper = hsv_rang_Seg[i+1] 
        mask = cv2.inRange(imageSegmented, lower, upper)
        if i==0:
            maskConcatenated = mask
        else:
            maskConcatenated = cv2.add(maskConcatenated, mask)
    
    return cv2.bitwise_and(imageSegmented, imageSegmented, mask = maskConcatenated)    
    
            
def get_templete_matching(imageSegmented):
    plt.imshow(imageSegmented)
    plt.show
    

def get_inside_grid_segmentation(x, y ,w, h, image, fullMask):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if (280 > w > 30) and (280  >h > 30) and aspect>0.75:
        
        imageSegmented = image[y:y+h,x:x+w]
        copyImageSeg = image[y:y+h,x:x+w]
        
        bitwiseRes = apply_mask_color_Inner(imageSegmented)
    
        greyRes  = cv2.cvtColor(bitwiseRes, cv2.COLOR_BGR2GRAY)
        fillRatioOnes = np.count_nonzero(greyRes)
        sizeMatrix = np.shape(greyRes)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        if fillRatio > 0.45:
            
#            get_templete_matching(copyImageSeg)

            ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
            fullMask[y:y+h,x:x+w] = thresh
            ret1, thresh1 = cv2.threshold(greyRes, 0, 255, cv2.THRESH_BINARY)
            
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),4)

            image[y:y+h,x:x+w, 0]  =  thresh1
            image[y:y+h,x:x+w, 1]  =  thresh1
            image[y:y+h,x:x+w, 2]  =  thresh1
#            get_templete_matching(copyImageSeg)
                

            
            

