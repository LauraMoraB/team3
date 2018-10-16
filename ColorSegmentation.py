from ImageFeature import get_full_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

hsv_rang = (
     np.array([0,150,50]), np.array([20, 255, 255]) #RED
     ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
     ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
)

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
    

def apply_color_mask(fullImage):
    
    size_hsv_rang = np.size(hsv_rang,0)
    for i in range(0, size_hsv_rang-1, 2):
        lower = hsv_rang[i]
        upper = hsv_rang[i+1] 
        
        mask = cv2.inRange(fullImage, lower, upper)
        if i==0:
            maskConcatenated = mask
        else:
            maskConcatenated = cv2.add(maskConcatenated, mask)
            
    mask = apply_morphology_operations(maskConcatenated)

    return cv2.bitwise_and(fullImage, fullImage, mask = mask)


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
    



def color_segmentation(df, path):

    listOfBB = []
    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        fullImage = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']   
        # Prepares mask files
        sizeImg  = np.shape(fullImage)     
        fullMask = np.zeros((sizeImg[0], sizeImg[1]))
        # Color space change

        fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV)
        rgbimg = cv2.cvtColor(fullImage, cv2.COLOR_HSV2RGB)
        
        bitwiseRes = apply_color_mask(fullImage)
        
        blur = cv2.blur(bitwiseRes, (5, 5), 0)            
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(bitwiseRes, [cnt], 0,255,-1)
                get_inside_grid_segmentation(x, y, w, h, rgbimg, fullMask, bitwiseRes, listOfBB, imageName[:-3])
                        
#        plt.imshow(fullMask)
#        plt.show()
#        plt.imshow(rgbimg)
#        plt.show()
        cv2.imwrite(path+'resultMask/mask.'+imageName[:-3]+'png', fullMask)
        cv2.imwrite(path+'resultMask/resBB.'+imageName[:-3]+'png', rgbimg)
    return listOfBB

        


            
def get_templete_matching(imageSegmented):
    plt.imshow(imageSegmented)
    plt.show
    

def get_inside_grid_segmentation(x, y ,w, h, image, fullMask, currentMask, listOfBB, imageName):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if (280 > w > 30) and (280  >h > 30) and aspect>0.75:
        
        imageSegmented = image[y:y+h,x:x+w]
        bitwiseResSegmented = currentMask[y:y+h,x:x+w]
#        bitwiseRes = apply_mask_color_Inner(imageSegmented)
    
        greyRes  = cv2.cvtColor(bitwiseResSegmented, cv2.COLOR_BGR2GRAY)
        fillRatioOnes = np.count_nonzero(greyRes)
        sizeMatrix = np.shape(greyRes)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        if fillRatio > 0.45:
            listOfBB.append((imageName,x,y,w,h))
#            get_templete_matching(copyImageSeg)

            ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
            fullMask[y:y+h,x:x+w] = thresh
#            ret1, thresh1 = cv2.threshold(greyRes, 0, 255, cv2.THRESH_BINARY)
            
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),10)
#
#            image[y:y+h,x:x+w, 0]  =  thresh1
#            image[y:y+h,x:x+w, 1]  =  thresh1
#            image[y:y+h,x:x+w, 2]  =  thresh1
#            get_templete_matching(copyImageSeg)
                
