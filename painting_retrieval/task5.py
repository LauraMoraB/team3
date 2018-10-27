import numpy as np
import pywt
import cv2
from skimage.feature import greycomatrix, greycoprops
from task1 import compute_histogram
from utils import plot_gray, get_image


def texture_method1(df, path):
    method_result = []
    for i in range(len(df)):
        dfSingle = df.iloc[i]
        name = dfSingle['Image']
        im = get_image(name, path).astype(np.uint8)
        gIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        plot_gray(gIm)
#        method_result.append((name, texture_region(gIm, method = 0, level_div = 0, level_wavelet = 0)[0]))
        texture_im = texture_region(gIm, method = 0, level_div = 0, level_wavelet = 0)
        method_result.append(texture_im[0].copy())
    return method_result
    
def haar_wavelet(im, level = 0):
# Computes the Haar Wavelet transform of a given imagen with dimension -> level
#                             -------------------
#                             |        |        |
#                             | cA(LL) | cH(LH) |
#                             |        |        |
# (cA, (cH, cV, cD))  <--->   -------------------
#                             |        |        |
#                             | cV(HL) | cD(HH) |
#                             |        |        |
#                             -------------------    
    
    cA, (cH, cV, cD) = pywt.dwt2(im, 'haar')
    if(level == 0):    
#        return cA, cH, cV, cD
        return [cA, cH, cV, cD]
    else:
        coeffCA = haar_wavelet(cA, level -1)
        return [coeffCA, cH, cV, cD]
    

def haar_sticking(coeff, level = 0):
# Sticks together wavelet transform coefficient to form a new image of original size    
    if(level == 0):
        return np.vstack((np.hstack((coeff[0],coeff[1])),np.hstack((coeff[2],coeff[3]))))       
    else:
        stickCA = haar_sticking(coeff[0], level - 1)
        stickCA = norm_im(stickCA, coeff[1])
        return np.vstack((np.hstack((stickCA, coeff[1])),np.hstack((coeff[2],coeff[3]))))    

def GLCM(im) :
# Computes Gray-Level Co-Ocurrence Matrix in four direction with 1 pixel distance
    # for each image 4 GLCM are computed at 0, 45, 90, -45 and -135º 
    imgStats = []
    glcm = greycomatrix(im, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    imgStats.append(np.mean(greycoprops(glcm, 'contrast')))
    imgStats.append(np.mean(greycoprops(glcm, 'energy')))
    imgStats.append(np.mean(greycoprops(glcm, 'correlation'))) 
    imgStats.append(np.mean(greycoprops(glcm, 'homogeneity')))      
    return glcm, imgStats


def texture_region(im, method = 0, level_div = 0, level_wavelet = 0):
    """
    im: image
    level_div: level of segmentation
    
    return: list of texture descriptors for the different image regions   
    """

    div = 2**level_div
       
    h, w = im.shape[0] , im.shape[1]
   
    w_step = int(w/div)
    h_step = int(h/div)
    
    return [compute_texture(im[y:y+h_step,x:x+w_step].copy(), method, level_wavelet) \
    for x in range(0,w,w_step) \
        if x+w_step <= w \
    for y in range(0,h,h_step) \
        if y+h_step<= h]


def compute_texture(im, method, level):
    coeff = haar_wavelet(im, level)
    result = []
    listCoeff = []
    listCoeff = list_detail_coeff(coeff.copy(), level, listCoeff)
    for component in listCoeff: 
        if(method == 0):
            result.append(compute_histogram(uint8_im(component), 0))
        elif(method == 1):
            result.append(GLCM(uint8_im(component)))        
    imgHaar = haar_sticking(coeff, level)
    plot_gray(imgHaar)
    return to_descriptor_format(result.copy())


def norm_im(im, reference):
    h, w = reference.shape[0] , reference.shape[1]
    return im[:h,:w]


def list_detail_coeff(coeff, level, listCoeff = []):
    firstCoeff = coeff.pop(0)
    listCoeff.extend(coeff)
    if(level == 0):
        return listCoeff
    else:
        listCoeff = list_detail_coeff(firstCoeff.copy(), level-1, listCoeff)    
        return listCoeff


def uint8_im(im):
    if(np.min(im)>=0):
        im = (im)/2       
    else:
        im = (im+256)/2
    image = im.astype(np.uint8)    
    return image
    

def to_descriptor_format(data):
    descriptor_list = []
    for list_hist in data:
        for hist in list_hist:
            descriptor_list += hist.astype(np.uint8).tolist()
    return descriptor_list


    
    
    