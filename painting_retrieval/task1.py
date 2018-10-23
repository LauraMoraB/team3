import cv2
import numpy as numpy
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

# pyramid images compression

# divide image in 4 zones
    