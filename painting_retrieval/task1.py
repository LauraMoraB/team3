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

# image white balance

# compute histogram
def compute_histogram(im, channel):
    """
    channel: must be 0,1 or 2
    """
    hist = cv2.calcHist([im], [channel], None, [256], [0,256])
    return hist

def histogram_region(im, channel, mask):
    """
    compute histogram of one of the regions in the image
    """
    hist = cv2.calcHist([im], [channel], mask, [256], [0,256])
    return hist

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
    portions=[]
    w, h = im.shape()
    w_small=w/division
    h_small=h/division
    
    for position in range(0, w, w_small) :
        portions.append(position)
        
        
    return portions

    