import cv2
import numpy as np
from utils import list_ds, get_gray_image, plot_matches

def compute_sift(path, rootSift = False, eps=1e-7):
    sift_result = []
    # Get DS images names list   
    im_list = list_ds(path)
    # Creates SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    for imName in im_list:
        # Load Gray version of each image
        imGray = get_gray_image(imName, path)
        # Find KeyLpoints and Sift Descriptors, info about KeyPoint objects -> https://docs.opencv.org/3.3.1/d2/d29/classcv_1_1KeyPoint.html
        (kps, descs) = sift.detectAndCompute(imGray, None)
        # In case no kps were found
        if len(kps) == 0:
            (kps, descs) = ([], None)
        # RootSift descriptor, sift improvement descriptor
        elif(rootSift == True):
            descs /= (descs.sum(axis=1, keepdims=True) + eps)
            descs = np.sqrt(descs)            
        # Append results
        sift_result.append([imName, kps, descs])    
    return sift_result

def BFMatcher(N, siftA, siftB, pathA = '', pathB = '', plot = False):
    imNameA, kpsA, descsA = siftA    
    imNameB, kpsB, descsB = siftB    
    # create a BFMatcher object which will match up the SIFT features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)    
    # Useful info about DMatch objects -> https://docs.opencv.org/java/2.4.9/org/opencv/features2d/DMatch.html
    matches = bf.match(descsA, descsB)
    # Sort the matches in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # keep N top matches
    matches = matches[0:N]
    if(plot == True):
        plot_matches(siftA, siftB, pathA, pathB, matches)        
    return matches