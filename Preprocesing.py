from ImageFeature import get_full_image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def apply_morphology_operations(mask):

    kernelOpen = np.ones((10,10),np.uint8)
    kernelClose = np.ones((10,10),np.uint8)
    kernelErode = np.ones((5,5),np.uint8)
    kernelDialte  = np.ones((10,10),np.uint8)
    
    mask = cv2.erode(mask ,kernelErode, iterations = 1)   
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    mask = cv2.dilate(mask,kernelDialte, iterations = 2)
    return mask
    

def apply_color_mask(fullImage):

    rojo_bajos1 = np.array([0,150,50], dtype=np.uint8) #RED
    rojo_altos1 = np.array([20, 255, 255], dtype=np.uint8) #RED
    
    blue_bajos1 = np.array([100,30,50], dtype=np.uint8) #BLUE
    blue_altos1 = np.array([150, 255, 255], dtype=np.uint8) #BLUE

    
    mascara_rojo1 = cv2.inRange(fullImage, rojo_bajos1, rojo_altos1)
    mascara_blue1 = cv2.inRange(fullImage, blue_bajos1, blue_altos1)

        
    maskConcatenated = cv2.add(mascara_rojo1, mascara_blue1)
            
    return maskConcatenated
    

def preproces_image(fullImage):
    hsv = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV )
    hsv_planes = cv2.split(hsv)
    #clipLimit limite de contraste - tileGridSize ventana 
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    # V equalyse luminance
    hsv_planes[2] = clahe.apply(hsv_planes[2])        
    hsv = cv2.merge(hsv_planes)
    
    bgr_ecualizado = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR ) 
        
    return bgr_ecualizado

def color_segmentation(df, path):
    listOfBB = []
    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        imgBGR = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']  
         # Prepares mask files
        bgr_ecualizado= preproces_image(imgBGR)
        #lowpass filter
        blur =cv2.bilateralFilter(bgr_ecualizado,5,100,100) 
#        gaussian_3 = cv2.GaussianBlur(blur, (9,9), 10.0)
#        unsharp_image = cv2.addWeighted(blur, 1.5, gaussian_3, -0.5, 0, blur)
#        imgBGR=unsharp_image
#        imgBGR=bgr_ecualizado
        imgBGR=blur
        cv2.imwrite(path+'Result_fullImage/'+imageName[:-3]+'png', imgBGR)
        
#        sizeImg  = np.shape(imgBGR)     
        # Color space change, it operates on HSV, RGB 
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        
        
        imgRGB = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)

        # Get mask, color + morphology         
        color_mask = apply_color_mask(imgHSV)
        morphology_mask = apply_morphology_operations(color_mask) 
        
        bitwiseRes = cv2.bitwise_and(imgHSV, imgHSV, mask = color_mask)        

        heir, contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         CCL labels 
        if contours:
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(bitwiseRes, [cnt], 0,(0,255,0),-1)
                #get_inside_grid_segmentation(x, y, w, h, imgRGB, bitwiseRes, listOfBB, imageName[:-3])
#                        get_inside_grid_segmentation(x, y, w, h, imgRGB, fullMask, bitwiseRes, listOfBB, imageName[:-3])

        create_maskFolders('train')
        create_maskFolders('validation') 
        
        cv2.imwrite(path+'resultMask/colorMask/mask.'+imageName[:-3]+'png', color_mask)
        cv2.imwrite(path+'resultMask/morphologyMask/mask.'+imageName[:-3]+'png', morphology_mask)
        cv2.imwrite(path+'resultMask/finalMask/mask.'+imageName[:-3]+'bitwiseRes.png', bitwiseRes)
        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path+'resultMask/resBB.'+imageName[:-3]+'png', imgBGR)
    
    return listOfBB



def get_inside_grid_segmentation(x, y ,w, h, image, currentMask, listOfBB, imageName):
    if w<h:
        aspect = w/h
    else:
        aspect = h/w
    if (280 > w > 30) and (280  >h > 30) and aspect>0.75:
        
#        bitwiseResSegmented = get_region_of_interest(x, y ,w, h, currentMask)
        bitwiseResSegmented = currentMask[y:y+h,x:x+w]
    
        fillRatioOnes = np.count_nonzero(bitwiseResSegmented)
        sizeMatrix = np.shape(bitwiseResSegmented)
        fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
        fillRatio = fillRatioOnes/fillRatioZeros
        if fillRatio > 0.45:
            listOfBB.append((imageName,x,y,w,h))
            #analisis del primer segmento (rectangulo contiene un conorno )
            get_templete_matching(x, y ,w, h, bitwiseResSegmented , image)

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
        
        