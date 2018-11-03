# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:10:27 2018

@author: Zaius
"""
import cv2
from utils import list_ds, get_bgr_image, plot_matches

def compute_hog(path, resize):
    hog_results = {}
    # Get DS images names list   
    im_list = list_ds(path)
    
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    
    # Creates HOG object
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    for imName in im_list:
        imBGR = get_bgr_image(imName, path, resize)
        
#        gradient = compute_gradient(imBGR)
        
        descriptor = hog.compute(imBGR)
        
        hog_results[imName] = [imName, [], descriptor]
    return hog_results

