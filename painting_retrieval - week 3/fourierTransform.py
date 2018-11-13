#conda install pyzmq=14.7.0


import cv2
import matplotlib.pyplot as plt
import numpy as np
from resizeImage import image_resize
from houghTrasnform import houghTrasnform, houghTrasnformGrouped
import matplotlib.pyplot as plt

def discriteFourierTransform(img):
    rows, columns = np.shape(img)
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(columns)
    
    imBorder = cv2.copyMakeBorder(img, 0, m - rows, 0, n - columns, cv2.BORDER_CONSTANT, value = 0)
    
    planes  = [np.float32(imBorder), np.zeros(imBorder.shape, np.float32)]
    complexI = cv2.merge(planes)  # Add to the expanded another plane with zeros
     
#    dft1 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
#    dft2 = cv2.dft(np.float32(imBorder), flags=cv2.DFT_COMPLEX_OUTPUT)
    cv2.dft(complexI, complexI)  # this way the result may fit in the source matrix
    cv2.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))


    cv2.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    
    magI = planes[0]
    
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv2.log(magI, magI)
    
    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    
    cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX) # Transform the matrix with float values into a
    plt.imshow(magI)
    plt.show()

    
    
def fastFourierTransfomr(img):
    f = np.fft.rfft(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    


def resize(img, sizeLimit = 500):
    (h, w) = img.shape[:2]
    if(h>w):
        if(h > sizeLimit):
            img = image_resize(img, height = sizeLimit)
    else:
        if(w > sizeLimit):
            img = image_resize(img, width = sizeLimit)
    return img
    
if __name__ == "__main__":
    pathQuery = "dataset/w5_devel_random/"
    
    imBGR = cv2.imread(pathQuery+"ima_000000.jpg")
    imResize = resize(imBGR, 1024)
    imGray = cv2.cvtColor(imResize, cv2.COLOR_BGR2GRAY)
    houghTrasnformGrouped(imGray)