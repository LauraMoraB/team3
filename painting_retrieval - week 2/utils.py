import pickle
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
from resizeImage import image_resize

def save_images(kp, imName, image):
    
    img_with_keypoints = cv2.drawKeypoints(image, kp, outImage=np.array([]), color=(0, 0, 255), 
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imwrite("keypointsIm/"+str(round(random.random(),4))+imName, img_with_keypoints)
    
    
# Returns list of images in a path
def list_ds(path_images):
    listImages =[]
    # Import Data from directories
    for filename in os.listdir(path_images):
        if filename.endswith(".jpg"):
            listImages.append(filename)
    return listImages

# Creates directories if not already created
def create_dir(pathSave):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

# Return numpy array with gray scale image		
def get_gray_image(im, path, resize = False, sizeLimit = 500):
    imBGR = get_bgr_image(im, path, resize, sizeLimit)
    return cv2.cvtColor(imBGR, cv2.COLOR_BGR2GRAY)

# Return numpy array with BGR scale image		
def get_bgr_image(im, path, resize = False, sizeLimit = 500):
    imBGR = cv2.imread(path+im)
    if(resize == True):
        (h, w) = imBGR.shape[:2]
        if(h>w):
            if(h > sizeLimit):
                imBGR = image_resize(imBGR, height = sizeLimit)
        else:
            if(w > sizeLimit):
                imBGR = image_resize(imBGR, width = sizeLimit)
    return imBGR

def plot_gray(im):
    plt.imshow(im, cmap='gray')
    plt.show()

def plot_rgb(im):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

# Plots Image + sift kps
def plot_sift(sift, path, resize = False):
    imName, kps, descs = sift    
    bgrGray = get_bgr_image(imName, path)
    imGray = get_gray_image(imName, path)
    imSift = cv2.drawKeypoints(imGray, kps, None)
    plot_rgb(bgrGray)  
    plot_gray(imGray)  
    plot_rgb(imSift)  

## Plots matches between two images
#def plot_matches_nearest(siftA, siftB, pathA, pathB, good_matches, resize = False, **draw_params):
#    imNameA, kpsA, descsA = siftA    
#    imNameB, kpsB, descsB = siftB  
#    imA = get_bgr_image(imNameA, pathA, resize)
#    imB = get_bgr_image(imNameB, pathB, resize)
#    
#    img_matches = np.empty((max(imA.shape[0], imB.shape[0]), imA.shape[1]+imB.shape[1], 3), dtype=np.uint8)
#    #cv2.drawMatches(imA, kpsA, imB, kpsB, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    img3 = cv2.drawMatchesKnn(imA,kpsA,imB,kpsB,good_matches,None,**draw_params)
#    plt.figure(figsize=(12,6))
#    plt.imshow(img_matches)
#    plot_rgb(img_matches)  
    
# Plots matches between two images
def plot_matches(siftA, siftB, pathA, pathB, matches, resize = False):
    imNameA, kpsA, descsA = siftA    
    imNameB, kpsB, descsB = siftB  
    imA = get_bgr_image(imNameA, pathA, resize)
    imB = get_bgr_image(imNameB, pathB, resize)
    match_img = cv2.drawMatches(
        imA, kpsA,
        imB, kpsB,
        matches, imB.copy(), flags=0)
    plt.figure(figsize=(12,6))
    plt.imshow(match_img)
    plot_rgb(match_img)    
 
# Save list in PKL format    
def save_pkl(list_of_list, path):
    create_dir(path)
    with open(path+'result.pkl', 'wb') as f:
        pickle.dump(list_of_list, f)
        
# Read pkl file GT and returns it as a list        
def get_query_gt(pkl_fle):
    resultList = []
    with open(pkl_fle, 'rb') as f:
        data = pickle.load(f)    
    for query in data:
        queryList = []
        for entry in query[1]:
            if(entry >= 0):
                queryList.append('ima_{:06d}.jpg'.format(entry))
            else:
                queryList.append(-1)
        resultList.append(queryList)
    return resultList

# Slice dictionary as list objects
def slice_dict(d1, no, nf):
    return dict(list(d1.items())[no:nf])

## Average precision at K calculation
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    
    result = score / min(len(actual), k)
#    print('APK:', result)
    return result

## Mean average precision at K calculation
def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
