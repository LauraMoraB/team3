# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:41:19 2018

@author: Zaius
"""
from utils import list_ds, get_gray_image
from skimage.feature import hog

    
    
def compute_HOG(image):
    ppc = 16
    orient = 8
    cell_block = (4, 4)
    return hog(image, orientations=orient, pixels_per_cell=(ppc,ppc),cells_per_block=cell_block,block_norm= 'L2',visualise=True)

        
def list_images(path, resize = False):
    hog_images = []
    hog_features = []
    im_list = list_ds(path)

    for imName in im_list:
        
        image = get_gray_image(imName, path, resize, 256)
        fd, hog_image= compute_HOG(image)
        hog_images.append(hog_image)
        hog_features.append(fd)
    
    return hog_images, hog_features
    