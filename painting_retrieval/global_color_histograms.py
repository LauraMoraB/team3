# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 19:14:35 2018
@author: gaby1
"""
from matplotlib import pyplot as plt
import numpy as np
import cv2
from utils import create_dir
import matplotlib
matplotlib.use('Agg')

def equalyse_luminance_image(fullImage,spaceType): #spaceType='HSV'
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


def low_filter_unsharp(img_bgr):
    gaussian_3 = cv2.GaussianBlur(img_bgr, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(img_bgr, 1.5, gaussian_3, -0.5, 0, img_bgr)
    return unsharp_image

def white_balance_LAB(fullImage, spaceType): # spaceType="LAB"
    result=changeSpaceColor(fullImage, spaceType)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def changeSpaceColor(imagen, spaceType):
    
    if spaceType == "HSV":
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        return hsv
    elif spaceType =="HLS":
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
    
    
def preproces_image(fullImage, pathprep_resultDS, imageName):
    create_dir(pathprep_resultDS+'equalyse_luminance/')
    create_dir(pathprep_resultDS+'low_filter_unsharp/')
    create_dir(pathprep_resultDS+'white_balance_LAB/')

    imeq=equalyse_luminance_image(fullImage,"HSV")
    cv2.imwrite(pathprep_resultDS+'equalyse_luminance/'+imageName[:-3]+'jpg', imeq)
    imlf=low_filter_unsharp(imeq)
    cv2.imwrite(pathprep_resultDS+'low_filter_unsharp/'+imageName[:-3]+'jpg', imlf)
    im_wb=white_balance_LAB(imlf,"LAB")
    cv2.imwrite(pathprep_resultDS+'white_balance_LAB/'+imageName[:-3]+'jpg', im_wb)
    return im_wb



    
def channel2list(image): 
   channel0, channel1, channel2 = image[:,:,0], image[:,:,1], image[:,:,2]
   return (image2list(channel0), image2list(channel1), image2list(channel2))

def image2list(image):
    imageL =  image.tolist()
    pxL = []
    for rows in imageL:
        for px in rows:
            pxL.append(px)
    return pxL

def get_px_one(imagen, spaceType):
    chimage= changeSpaceColor(imagen, spaceType)
    channel0L, channel1L, channel2L=channel2list(chimage)
 
    imageL = image2list(imagen)

    validch0 = []
    validch1 = []
    validch2 = []
    pxCount = 0
    for px in imageL:
        if any(px):
            validch0.append(channel0L[pxCount])
            validch1.append(channel1L[pxCount])
            validch2.append(channel2L[pxCount])
        pxCount += 1
    return (validch0, validch1, validch2)

def global_color(fullImage, spaceType, pathprep_resultDS, imageName):
    create_dir(pathprep_resultDS+'Final/')
    
    im_prep=preproces_image(fullImage, pathprep_resultDS,imageName)
    im_ch=changeSpaceColor(im_prep, spaceType)
    cv2.imwrite(pathprep_resultDS+'Final/'+imageName, im_ch)
    return im_ch
    
def global_color_hist(fullImage, spaceType_hist, pathprep_resultDS, imageName):  
    imcolor=global_color(fullImage, spaceType_hist,pathprep_resultDS, imageName)
    (channel0Single, channel1Single, channel2Single) = get_px_one(imcolor, spaceType_hist)    
    return (channel0Single, channel1Single, channel2Single)

def save_global_color_hist(channel0Single, channel1Single, channel2Single, dfSingle,spaceType_hist, imageName, pathResults):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot([1,2,3])
    plt.hist(channel0Single, bins = 25, color= None)
    plt.ylabel('f')  
    plt.xlabel('channel0')  
    plt.title('channel0'+dfSingle)
    fig.savefig(pathResults+'Histograma_channel0_'+spaceType_hist+imageName)
    plt.close(fig)
   
    fig2=plt.figure()
    ax=fig2.add_subplot(111)
    ax.plot([1,2,3])
    plt.hist(channel1Single, bins = 25, color= None)
    plt.ylabel('f')  
    plt.xlabel('channel1')  
    plt.title('channel1'+dfSingle)
    fig2.savefig(pathResults+'Histograma_channel1_'+spaceType_hist+imageName)
    plt.close(fig2)

    fig3=plt.figure()
    ax=fig3.add_subplot(111)
    ax.plot([1,2,3])
    plt.hist(channel1Single, bins = 25, color= None)
    plt.ylabel('f')  
    plt.xlabel('channel2')  
    plt.title('channel2'+dfSingle)
    fig2.savefig(pathResults+'Histograma_channel2_'+spaceType_hist+imageName)
    plt.close(fig3)