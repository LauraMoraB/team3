import numpy as np
import pywt
from skimage.feature import greycomatrix, greycoprops



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
        coeffCH = haar_wavelet(cH, level -1)
        coeffCV = haar_wavelet(cV, level -1)
        coeffCD = haar_wavelet(cD, level -1)
        return coeffCA, coeffCH, coeffCV, coeffCD
    
# Stick together wavelet transform coefficient to form a new image of original size    
def haar_sticking(coeff, level = 0):
    if(level == 0):
        return np.vstack((np.hstack((coeff[0],coeff[1])),np.hstack((coeff[2],coeff[3]))))       
    else:
        stickCA = haar_sticking(coeff[0], level - 1)
        stickCH = haar_sticking(coeff[1], level - 1)
        stickCV = haar_sticking(coeff[2], level - 1)
        stickCD = haar_sticking(coeff[3], level - 1)
        return np.vstack((np.hstack((stickCA, stickCH)),np.hstack((stickCV, stickCD))))      

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
    
