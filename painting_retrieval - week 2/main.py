import cv2
import numpy as np
from utils import list_ds, save_pkl, mapk, create_dir, get_query_gt, plot_gray, plot_rgb, plot_sift, plot_matches, get_gray_image
import random

def init():
    # --> BEGINING FOLDERS PREPARATION <-- #
    paths = {}
    # Images Path --> Add new entry in the dictionary for new subfodlers!
    paths['pathDS'] = "dataset/"
    paths['pathQueriesValidation'] = "queries_validation/"
    paths['pathGTValidation'] = "queries_validation/GT/"
    paths['pathQueriesTest'] = "queries_test/"
    paths['pathGTTest'] = "queries_test/GT/"
    # General Results Path
    paths['pathResult'] = "results/"
    # Delivery Results Path
    paths['pathResultsM1'] = "results/method1/"
    paths['pathResultsM2'] = "results/method2/"
    paths['pathResultsM3'] = "results/method3/"
    
    # Create all subdirectories on dictionary if tey dont already
    for path in paths:
        create_dir(paths[path])
    print('All subfolders have been created')
    # --> END FOLDERS PREPARATION <-- #
    return paths

def load_sift_descriptors(path, rootSift = False, eps=1e-7):
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

def demo():
    # Example for ploting a sift image
    print('Sift kps example on random image from ds:')
    siftA = sift_ds[random.randint(0,len(sift_ds))]
    plot_sift(siftA, paths['pathDS'])
    print('Sift matching example on random image from ds:')
    siftA = sift_ds[0]
    siftB = sift_ds[0]
    BFMatcher(50, siftA, siftB, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True)   
    
if __name__ == "__main__":

    # Prepares folders
    paths = init()
    # Creates list of list with sift kps and descriptors  -> [imName, kps, descs]
    sift_ds = load_sift_descriptors(paths['pathDS'])
    sift_validation = load_sift_descriptors(paths['pathQueriesValidation'])
    # Demmonstration of status
    demo()
