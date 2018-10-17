from ImageFeature import get_full_image
import numpy as np
import cv2


def preproces_image(df, path):
    kernel_1 = np.ones((10,10),np.uint8)
    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        fullImage = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image'] 
        #change space to HSV to equalyse      
        hsv = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV )
        hsv_planes = cv2.split(hsv)
        #clipLimit limite de contraste - tileGridSize ventana 
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        # V equalyse luminance
        hsv_planes[2] = clahe.apply(hsv_planes[2])        
        hsv = cv2.merge(hsv_planes)
        
        bgr_ecualizado = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR )    
        
        #need \datasets\split\validation\bgr_ecualizado
        cv2.imwrite(path+'bgr_ecualizado/'+imageName[:-3]+'png', bgr_ecualizado)

        #d=9 sigmaColor =75 sigmaSpace =75
        #blur = cv2.bilateralFilter(bgr_ecualizado,5,100,100)
        fullImage =cv2.GaussianBlur(bgr_ecualizado,(21,21),0)        
        fullImage = cv2.bilateralFilter(bgr_ecualizado,5,100,100)

        #need \datasets\split\validation\fullImage_blur
        cv2.imwrite(path+'fullImage_blur/'+imageName[:-3]+'png', fullImage)
        # Prepares mask files
        sizeImg  = np.shape(fullImage)     
        fullMask = np.zeros((sizeImg[0], sizeImg[1]))
        # Color space change
        fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2HSV)
        rojo_bajos1 = np.array([2,50,50], dtype=np.uint8) #RED
        rojo_altos1 = np.array([15, 255, 255], dtype=np.uint8) #RED
        
        blue_bajos1 = np.array([100,30,30], dtype=np.uint8) #BLUE
        blue_altos1 = np.array([150, 255, 255], dtype=np.uint8) #BLUE

        
        mascara_rojo1 = cv2.inRange(fullImage, rojo_bajos1, rojo_altos1)
        mascara_rojo1 = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel_1)
                    
        mascara_blue1 = cv2.inRange(fullImage, blue_bajos1, blue_altos1)
        mascara_blue1 = cv2.morphologyEx(mascara_blue1, cv2.MORPH_CLOSE, kernel_1)

        
        fullMask = cv2.add(mascara_rojo1, mascara_blue1)
        bitwiseRes = cv2.bitwise_and(fullImage, fullImage, mask = fullMask)
        fullMask = cv2.morphologyEx(bitwiseRes, cv2.MORPH_DILATE, kernel_1)
        
        cv2.imwrite(path+'resultMask/mask.'+imageName[:-3]+'png', fullMask)
        
        #need \datasets\split\validation\Result_color_mask and show in HSV
        cv2.imwrite(path+'Result_color_mask/'+imageName[:-3]+'png', (bitwiseRes.astype('uint8')) * 255)