import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import get_image

# image equalization
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


# compute histogram
def compute_histogram(im, channel, mask=None,bins=256):
    """
    channel: must be 0, 1 or 2
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
       
    w, h = im.shape[1] , im.shape[0]
    
    w_step = int(w/div)
    h_step = int(h/div)
    
    return [compute_histogram(im[y:y+h_step,x:x+w_step], channel) \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]



def divide_image(im, div):
    """
    im: image
    div: number of regions per absis
    
    return: list of [y1, x1, y2, x2] from the different image regions 
    """
    w, h = im.shape[1] , im.shape[0]
    
    w_step = int(w/div)
    h_step = int(h/div)

    return [[y,x,y+h_step,x+w_step] \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]

    