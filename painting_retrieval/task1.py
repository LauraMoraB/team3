# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:34:18 2018

@author: Gaby
"""
from utils import get_full_image, get_image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')



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
        
        ch_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR ) 
        
    return ch_image
    
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

def get_px_values_cut(dfSingle, path, spaceType):
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

def get_px_one(imagen, spaceType):
    # return hue and sat values for valid px      
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
        dfSingle = df.iloc[0]
        imageName = dfSingle['Image']  

        (channel0Single, channel1Single, channel2Single) = get_px_values(dfSingle, path, spaceType)
        
        channel0L.extend(channel0Single)
        channel1L.extend(channel1Single)
        channel2L.extend(channel2Single)
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot([1,2,3])
        plt.hist(channel0Single, bins = 25, color = colors.pop(0))
        plt.ylabel('f')  
        plt.xlabel('channel0')  
        plt.title('channel0'+dfSingle)
        fig.savefig('Histograma/'+imageName)
#        imhistograma=cv2.imread('channel0.png',1) 
#        cv2.imwrite('Cambios/imhistograma'+imageName[:-3]+'png', imhistograma)



#    plt.hist(channel0L, bins = 25, color = colors.pop(0))
#    plt.ylabel('f')
#    plt.xlabel('channel0')
#    plt.title('channel0L')
#    plt.show()
    
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
    
def white_balance_LAB(fullImage, spaceType):
    result=changeSpaceColor(fullImage, spaceType)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def color_characterization(df, path, spaceType_prep, spaceType_hist):
    #♣colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
    channel0L=[]
    channel1L=[]
    channel2L=[]

    for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        imgBGR = get_full_image(dfSingle, path)    
        imageName = dfSingle['Image']  
         # Prepares mask files
        bgr_ecualizado= preproces_image(imgBGR, spaceType_prep)
        cv2.imwrite('Cambios/bgr_ecualizado_'+imageName[:-3]+'png', bgr_ecualizado)

        #lowpass filter
#        blur =cv2.bilateralFilter(bgr_ecualizado,5,100,100) 
        gaussian_3 = cv2.GaussianBlur(bgr_ecualizado, (9,9), 10.0)
        unsharp_image = cv2.addWeighted(bgr_ecualizado, 1.5, gaussian_3, -0.5, 0, bgr_ecualizado)
        imgBGR=unsharp_image
#        imgBGR=bgr_ecualizado
#        imgBGR=blur
        cv2.imwrite('Cambios/unsharp_image_'+imageName[:-3]+'png', unsharp_image)
        im_balance=white_balance_LAB(imgBGR,"LAB")
        cv2.imwrite('Cambios/im_balance_'+imageName[:-3]+'png', im_balance)
# Gets images one by one

        (channel0Single, channel1Single, channel2Single) = get_px_one(im_balance, spaceType_hist)
        
        channel0L.extend(channel0Single)
        channel1L.extend(channel1Single)
        channel2L.extend(channel2Single)
        
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot([1,2,3])
        plt.hist(channel0Single, bins = 25, color= None)
        plt.ylabel('f')  
        plt.xlabel('channel0')  
        plt.title('channel0'+dfSingle)
        fig.savefig('Cambios/Histograma_'+spaceType_hist+imageName)
        plt.close(fig)
        
        
def equalize_3_channels(colorIm):
    """
    Image equalization for three channels
    
    Return: equalized image
    """
    for c in range(0, 2):
       colorIm[:,:,c] = cv2.equalizeHist(colorIm[:,:,c])
       
    return colorIm

def equalize_1_channel(grayIm):
    grayIm = cv2.equalizeHist(grayIm)
    return grayIm

# image white balance

# compute histogram
def compute_histogram(im, channel,mask=None,bins=256):
    """
    channel: must be 0,1 or 2
    """
    hist = cv2.calcHist([im], [channel], mask, [bins], [0,bins])
    return hist

def histogram_region(im, channel, level):
    """
    im: image
    level: level of segmentation
    channel: 0,1,2
    
    return: list of histograms from the different image regions 
    hist_channel = [ [hist_region1], [hist_region2],.., [hist_regionN] ]
    
    """
    div = 2**level
       
    w, h = im.shape[0] , im.shape[1]
    
    w_step = int(w/div)
    h_step = int(h/div)
    
    return [compute_histogram(im[y:y+h_step,x:x+w_step], channel) \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]

# pyramid images compression
# Gaussian pyramid
def gaussian_pyramid(im, levels):
    G = im.copy()
    pyramid = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        pyramid.append(G)
        
    return pyramid

# divide image in 4 zones
def divide_image(im, division):
    
    w = im.shape[0]
    h = im.shape[1]

    
    w_step = int(w/div)
    h_step = int(h/div)

    return [[y,x,y+h_step,x+w_step] \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]
