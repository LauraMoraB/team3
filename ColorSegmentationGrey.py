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
            

    return maskConcatenated



def color_segmentation_grey(df, path):

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
        
        maskColor = apply_color_mask(fullImage)
        maskMorph = apply_morphology_operations(maskColor)
        bitwiseRes = cv2.bitwise_and(fullImage, fullImage, mask = maskMorph)
        
        blur = cv2.blur(bitwiseRes, (5, 5), 0)            
        imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
        heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        txtFile = open("bbresults/grey/gt."+imageName[:-3]+"txt", "w")

        if contours:
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(bitwiseRes, [cnt], 0,255,-1)
                get_inside_grid_segmentation(x, y, w, h, rgbimg, fullMask, bitwiseRes, txtFile)
                
    
        txtFile.close()
        
#        plt.imshow(fullImage)
#        plt.show()
#        plt.imshow(maskColor)
#        plt.show()
#        plt.imshow(maskMorph)
#        plt.show()
#        plt.imshow(bitwiseRes)
#        plt.show()
#        plt.imshow(fullMask)
#        plt.show()
        plt.imshow(rgbimg)
        plt.show()
        cv2.imwrite(path+'resultMask/grey/mask.'+imageName[:-3]+'png', fullMask)
#        cv2.imwrite(path+'resultMask/resBB.'+imageName[:-3]+'png', rgbimg)


def get_inside_grid_segmentation(x, y ,w, h, image, fullMask, currentMask, txtFile):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if (280 > w > 30) and (280  >h > 30) and aspect>0.65:
        
#        bitwiseResSegmented = get_region_of_interest(x, y ,w, h, currentMask)
        bitwiseResSegmented = currentMask[y:y+h,x:x+w]
        BGRSegmented = image[y:y+h,x:x+w]
        greyResImage = cv2.cvtColor(BGRSegmented, cv2.COLOR_BGR2GRAY)
    
        greyRes  = cv2.cvtColor(bitwiseResSegmented, cv2.COLOR_BGR2GRAY)
        fillRatioOnes = np.count_nonzero(greyRes)
        sizeMatrix = np.shape(greyRes)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        if fillRatio > 0.35:

            ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
            fullMask[y:y+h,x:x+w] = thresh
            tipo = get_templete_matching(x, y ,w, h, greyResImage, image)
            
            txtFile.write(str(float(y))+" "+ str(float(x))+ " " + str(float(h+y)) + " " + str(float(w+x)) + " " + tipo + "\n")

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
    maximumFinal= 0;
    tipo = "NULL"
    
    for i in range(1,7):
        maskTemplate = cv2.imread("template/avgGrey/temp"+str(i)+".png",0)
        maskTemplate = cv2.resize(maskTemplate,(w-1,h-1))
        # Perform match operations. 
        res = cv2.matchTemplate(maskSegmented, maskTemplate, cv2.TM_CCOEFF_NORMED) 
          
        # Specify a threshold 

        threshold = 0.6
        
        maximumLocal = np.max(res)
        if (maximumLocal>=threshold and maximumLocal> maximumFinal):
            maximumFinal = maximumLocal
            # Store the coordinates of matched area in a numpy array 
            loc = np.where( res >= threshold)  
            
            # Draw a rectangle around the matched region. 
            for pt in zip(*loc[::-1]): 
    #            cv2.rectangle(image, (pt[0]+x, pt[1]+y), (pt[0] + x + w, pt[1] + y + h), (0,255,0), 5)
                if(i==1):
                    cv2.putText(image, 'A', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "A"
                elif(i==2):
                    cv2.putText(image, 'B', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "B"
                elif(i==3 ):
                    cv2.putText(image, 'C', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "C"
                elif(i==4 ):
                    cv2.putText(image, 'D', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "D"
                elif(i==5 ):
                    cv2.putText(image, 'E', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "E"
                else:
                    cv2.putText(image, 'F', (pt[0]+x, pt[1]+y),cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                    tipo = "F"
    return tipo



