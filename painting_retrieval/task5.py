import numpy as np
import pywt
from skimage.feature import greycomatrix, greycoprops
from task1 import compute_histogram
from utils import plot_gray



#                             -------------------
#                             |        |        |
#                             | cA(LL) | cH(LH) |
#                             |        |        |
# (cA, (cH, cV, cD))  <--->   -------------------
#                             |        |        |
#                             | cV(HL) | cD(HH) |
#                             |        |        |
#                             -------------------



# Computes the Haar Wavelet transform of a given imagen with dimension -> level
def haar_wavelet(img, level = 0):
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    if(level == 0):    
        return cA, cH, cV, cD
    else:
        coeffCA = haar_wavelet(cA, level -1)
        return coeffCA, cH, cV, cD
    
# Stick together wavelet transform coefficient to form a new image of original size    
def haar_sticking(coeff, level = 0):
    if(level == 0):
        return np.vstack((np.hstack((coeff[0],coeff[1])),np.hstack((coeff[2],coeff[3]))))       
    else:
        stickCA = haar_sticking(coeff[0], level - 1)
        stickCA = norm_im(stickCA, coeff[1])
        return np.vstack((np.hstack((stickCA, coeff[1])),np.hstack((coeff[2],coeff[3]))))    
# Computes Gray-Level Co-Ocurrence Matrix in four direction with 1 pixel distance
def GLCM(img) :
    # for each image 4 GLCM are computed at 0, 45, 90, -45 and -135º 
    imgStats = []
    glcm = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    imgStats.append(np.mean(greycoprops(glcm, 'contrast')))
    imgStats.append(np.mean(greycoprops(glcm, 'energy')))
    imgStats.append(np.mean(greycoprops(glcm, 'correlation'))) 
    imgStats.append(np.mean(greycoprops(glcm, 'homogeneity')))      
    return glcm, imgStats


def texture_region(im, level_div = 0, level_wavelet = 0):
    """
    im: image
    level: level of segmentation
    
    return: list of texture descriptors for the different image regions   
    """

    div = 2**level_div
       
    h, w = im.shape[0] , im.shape[1]
   
    w_step = int(w/div)
    h_step = int(h/div)
    
    return [compute_texture(im[y:y+h_step,x:x+w_step], level_wavelet) \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]

def compute_texture(im, level):
    coeff = haar_wavelet(im, level)
    imgHaar = haar_sticking(coeff, level)
    plot_gray(imgHaar)
    return None

def norm_im(im, reference):
    h, w = reference.shape[0] , reference.shape[1]
    return im[:h,:w]