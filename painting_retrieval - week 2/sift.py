import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import list_ds, get_gray_image, plot_matches

def compute_sift(path, rootSift = False, eps=1e-7):
    sift_result = {}
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
        sift_result[imName] = [imName, kps, descs] 
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
        # Plots both images + theirs coincident matches
        plot_matches(siftA, siftB, pathA, pathB, matches)        
    return matches


def retreive_image(siftDs, siftQueries, paths, k, th = 60, descsMin = 3):   
    queriesResult = []
    distancesResult = []
    for imNameQuery in siftQueries:
        matches = []
        siftQuery = siftQueries[imNameQuery]
        for imNameDs in siftDs:
            siftIm = siftDs[imNameDs]
            # As a default just return 100 best matches per image, could be incresed
            matchesBF = BFMatcher(100, siftQuery, siftIm, pathA=paths['pathQueriesValidation'], 
                                  pathB = paths['pathDS'], plot = True)  
            distance = [o.distance for o in matchesBF if o.distance <= th]
            # if less than descsMin matches found, not considered a match
            if(len(distance) >= descsMin):
                matches.append([imNameDs, distance])
        # Sort images per number of matches under threshold level
        matches = sorted(matches, key = lambda x:len(x[1]), reverse = True)
        # if more than K matches, return the better K
        if(len(matches) > k):
            matches = matches[0:k] 
        # Contruct query result to be returend
        distancesResult.append([l[1] for l in matches ])
        queriesResult.append([l[0] for l in matches ])
            
    return queriesResult, distancesResult

# Computes distances taking into account GT pairs
def get_gt_distance(N, sift_ds, sift_validation, gt_list, paths):   
    i = 0
    validationMatches = []
    for imName in sift_validation:
        siftA = sift_validation[imName]
        siftB = sift_ds[gt_list[i][0]]
        matchesBF = BFMatcher(N, siftA, siftB, pathA=paths['pathQueriesValidation'], 
                              pathB = paths['pathDS'], plot = True)  
        distance = [o.distance for o in matchesBF]
        validationMatches.append([imName, distance])
        i += 1
    return validationMatches

# Creates Stats from Distances Results
def get_distances_stats(N, matches, plot = False):
    distances = []
    for n in range(N):
        for i in range(len(matches)):
            try:
                if(i == 0):
                    distances.append([matches[i][1][n]])
                else:
                    distances[n].append(matches[i][1][n])
            except IndexError:
                    if(i == 0):
                        distances.append([None])
                    else:
                        distances[n].append(None)
    stats = []
    for entry in distances:
        try:
            entry = [x for x in entry if x != None]
        except ValueError:
            pass 
        stats.append([np.min(entry),np.max(entry),np.mean(entry),np.std(entry)])

    result = np.array(stats)
    if(plot == True):
        plt.errorbar(np.arange(N), result[:,2], result[:,3], fmt='ok', lw=3)
        plt.errorbar(np.arange(N), result[:,2], [result[:,2] - result[:,0], 
                     result[:,1] - result[:,2]],fmt='.k', ecolor='gray', lw=1)
        plt.xlim(-1, N)   
        plt.ylabel('Distance')
        plt.xlabel('Ordered descritor #')
        plt.title('GT BFMatcher')
        plt.show()        
        
    return result