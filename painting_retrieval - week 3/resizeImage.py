import cv2
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized


def image_save_original_sizes(imgOrig, imgDest):
    [orig_h, orig_w] = np.shape(imgOrig)
    [dest_h, dest_w] = np.shape(imgDest)
    
    percent_w =  orig_w / dest_w
    precent_h =  orig_h / dest_h
    
    return percent_w, precent_h

        
