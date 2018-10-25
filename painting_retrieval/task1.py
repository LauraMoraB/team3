# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:34:18 2018

@author: Gaby
"""
from utils import get_full_image
from matplotlib import pyplot as plt
import numpy as np
import cv2

def compute_normColor(im, channel):
    # convert input image to the normRGB color space

    normrColor_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:,:,0] + im[:,:,1] + im[:,:,2] + eps_val

    normrColor_im[:,:,0] = im[:,:,0] / norm_factor_matrix
    normrColor_im[:,:,1] = im[:,:,1] / norm_factor_matrix
    normrColor_im[:,:,2] = im[:,:,2] / norm_factor_matrix
    
    # Develop your method here:
    # Example:
    if normrColor_im=='0':
    
        pixel_candidates = normrColor_im[:,:,0]>100;
    if normrColor_im=='1':    
        pixel_candidates = normrColor_im[:,:,1]>100;

    if normrColor_im=='2':    
        pixel_candidates = normrColor_im[:,:,2]>100;  

    return pixel_candidates
 

def preproces_image(fullImage,spaceType):
    ch_image=changeSpaceColor(fullImage, spaceType)
    if(spaceType=="HSV"):
        hsv_planes = cv2.split(ch_image)
        #clipLimit limite de contraste - tileGridSize ventana 
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        # V equalyse luminance
        hsv_planes[2] = clahe.apply(hsv_planes[2])        
        hsv = cv2.merge(hsv_planes)
        
        bgr_ecualizado = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR ) 
        
    return bgr_ecualizado
    
def image2list(image):
    # convert image array into unidimensional(pixel) ordered list
    imageL =  image.tolist()
    pxL = []
    for rows in imageL:
        for px in rows:
            pxL.append(px)
    return pxL
def channel2list(image): #########################################################################Explicacion 
    # converts RGB image array into H, S, V, unidimensional ordered list
   channel0, channel1, channel2 = image[:,:,0], image[:,:,1], image[:,:,2]
   return (image2list(channel0), image2list(channel1), image2list(channel2))

           
def get_px_values(dfSingle, path, spaceType):
    # return hue and sat values for valid px  
    imagen = get_full_image(dfSingle, path)

    chimage= changeSpaceColor(imagen, spaceType)
    channel0L, channel1L, channel2L=channel2list(chimage)
 
    imageL = image2list(imagen)

    validch0 = []
    validch1 = []
    validch2 = []
    pxCount = 0
    for px in imageL:
        # (px)
        if any(px):
            validch0.append(channel0L[pxCount])
            validch1.append(channel1L[pxCount])
            validch2.append(channel2L[pxCount])
        pxCount += 1
    return (validch0, validch1, validch2)


def compute_histograms(df, path, spaceType):
    
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
    channel0L=[]
    channel1L=[]
    channel2L=[]

    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        (channel0Single, channel1Single, channel2Single) = get_px_values(dfSingle, path, spaceType)
        
        channel0L.extend(channel0Single)
        channel1L.extend(channel1Single)
        channel2L.extend(channel2Single)
    
    plt.hist(channel0L, bins = 25, color = colors.pop(0))
    plt.ylabel('f')
    plt.xlabel('channel0')
    plt.title('channel0L')
    plt.show()
    
#    plt.hist(channel1L, bins = 25, color = colors.pop(0))
#    plt.ylabel('f')
#    plt.xlabel('sat')
#    plt.title('channel1 ')
#    plt.show()
#    
#    plt.hist(channel2L, bins = 25, color = colors.pop(0))
#    plt.ylabel('f')
#    plt.xlabel('val')
#    plt.title('channel2 ')
#    plt.show()
        

def changeSpaceColor(imagen, spaceType):
#SOLO CAMBIO DE ESPACIO DE COLO LA IMAGEN ORIGINAL RECORTADA    
    if spaceType == "HSV":
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        return hsv
    elif spaceType =="HLS":
#        hls = cv2.cvtColor(imagen, cv2.COLOR_RGB2HLS )
        hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS )
        return hls
    
    elif spaceType == "LAB":   
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        return lab

    elif spaceType == "YCrCb":   
        ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
        return ycrcb

    elif spaceType == "XYZ":   
        xyz = cv2.cvtColor(imagen, cv2.COLOR_BGR2XYZ)
        return xyz
    elif spaceType == "LUV":   
        Luv = cv2.cvtColor(imagen, cv2.COLOR_BGR2Luv )
        return Luv
    else:
        return imagen
    
def color_characterization(df, path):

   for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        imgBGR = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']  
         # Prepares mask files
        bgr_ecualizado= preproces_image(imgBGR, 'HSV')
        cv2.imwrite('Cambios/bgr_ecualizado_'+imageName[:-3]+'png', bgr_ecualizado)

        #lowpass filter
#        blur =cv2.bilateralFilter(bgr_ecualizado,5,100,100) 
        gaussian_3 = cv2.GaussianBlur(bgr_ecualizado, (9,9), 10.0)
        unsharp_image = cv2.addWeighted(bgr_ecualizado, 1.5, gaussian_3, -0.5, 0, bgr_ecualizado)
        imgBGR=unsharp_image
#        imgBGR=bgr_ecualizado
#        imgBGR=blur
        cv2.imwrite('Cambios/unsharp_image_'+imageName[:-3]+'png', unsharp_image)
   
 
#        heir, contours, _ = cv2.findContours(morphology_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        if contours:
#            for cnt in contours:
#                x,y,w,h = cv2.boundingRect(cnt)
#                cv2.drawContours(bitwiseRes, [cnt], 0,255,-1)
#        
#        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
#        cv2.imwrite(path+'resultMask/resBB.'+imageName[:-3]+'png', imgBGR)
