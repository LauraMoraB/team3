import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import list_ds, get_gray_image, plot_matches, save_images


def compute_threshold(matcherType, method, ROOTSIFT):
    if matcherType == "BFMatcher":
        
        if method == "SIFT":
            if ROOTSIFT == False:
                th = 90 
                # Min number of matches to considerer a good retrieval
                descsMin = 15
            else:
                th = 0.15
                descsMin = 5
                
        elif method=="ORB":
            th = 25
            descsMin = 5
            
        elif method=="KAZE":
            th = 0.3
            descsMin = 5
        elif method =="HOG":
            th = 1
            descsMin = 5
        elif method=="FREAK":
            th = 25
            descsMin = 5

        elif method=="SURF":
            th = 0.35
            descsMin = 5
            
        else: 
            print("invalid method: ", method)
            
    # if Flann       
    else:
        if method=="DAISY":
            th = 0.5
            descsMin = 3
            
        elif method == "SIFT":
            # valor distancia min
            th = 0.7 
            # Min number of matches to considerer a good retrieval
            descsMin = 50
            
        elif method=="ORB":
            th = 0.5
            descsMin = 5
            
        elif method=="KAZE":
            th = 0.3
            descsMin = 5

        elif method=="SURF":
            th = 0.35
            descsMin = 5

            
        elif method =="HOG":
            th = 0.3
            descsMin = 5
        else: 
            print("invalid method: ", method)
        
    return th, descsMin

def feature_detection(featureType, im):
    
    if featureType=="SURF":
        minHessian = 100
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
        return detector.detect(im)
        
    elif featureType=="FAST":
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        return fast.detect(im,None)
    

def compute_kp_desc(im, method, descriptor):
    """
    Compute kps and desc depending of the chosen method
    """
    
    if method == "DAISY" or method=="FREAK":
        # FAST / SURF
        keypoints = feature_detection("SURF", im)
        desc = descriptor.compute(im, keypoints)   
        return keypoints, desc
    
    elif method == "HOG":
        locs = []

        ders = descriptor.compute(im)
        return (locs, ders)
    
    else:
        return descriptor.detectAndCompute(im, None) 
        

def init_method(method):
    if method == "SIFT":
        return cv2.xfeatures2d.SIFT_create()
    
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=500,scoreType=cv2.ORB_HARRIS_SCORE)
    
    elif method == "HOG":
        winSize = (64,64)
        blockSize = (32,32)
        blockStride = (16,16)
        cellSize = (16,16)
        nbins = 9
        return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins )

    elif method == "DAISY":
        return cv2.xfeatures2d.DAISY_create()

    elif method == "FREAK":
        return cv2.xfeatures2d.FREAK_create()
    elif method=="KAZE":
        return cv2.KAZE_create(extended=True, upright=True, threshold=0.001)
    
    elif method == "SURF":
        return cv2.xfeatures2d.SURF_create(1200)

    
def define_measurement(method):
    
    if method == "SIFT":
        return cv2.NORM_L2
    
    elif method == "ORB":
        return cv2.NORM_HAMMING
    
    elif method == "SURF": #set to NORM_L1 or NORM_L2.
        return cv2.NORM_L1
    
    elif method == "HOG":
        return cv2.NORM_L2
    
    else: 
        return cv2.NORM_L2
        


def define_prepared_image(method, imName, path, resize):
    if method == "HOG":
        return get_gray_image(imName, path, resize, 256)
    else:
        return get_gray_image(imName, path, resize)

    
    
def compute_sift(path, method, resize = False, rootSift = False, eps = 1e-7, save = False):
    sift_result = {}
    # Get DS images names list   
    im_list = list_ds(path)
    
    # Creates SIFT object
    desc_init = init_method(method)

    for imName in im_list:
        print(imName)
        # Load Gray version of each image
        imSource = define_prepared_image(method, imName, path, resize)
        
        # Find KeyLpoints and Sift Descriptors, info about KeyPoint objects -> https://docs.opencv.org/3.3.1/d2/d29/classcv_1_1KeyPoint.html
        (kps, descs) = compute_kp_desc(imSource, method, desc_init)
        
        if save == True:
            save_images(kps, imName, imSource)
        
        if method == 'HOG':
            descs = descs.ravel()
        # In case no kps were found
        elif len(kps) == 0:
            (kps, descs) = ([], None)
            
        # RootSift descriptor, sift improvement descriptor
        elif(rootSift == True):
            descs /= (descs.sum(axis=1, keepdims=True) + eps)
            descs = np.sqrt(descs) 
            
        # Append results        
        sift_result[imName] = [imName, kps, descs] 
    
    return sift_result

def BFMatcher(N, siftA, siftB, method, pathA = '', pathB = '', plot = False, resize = False):
    imNameA, kpsA, descsA = siftA    
    imNameB, kpsB, descsB = siftB   
    
    # fix how data is stored
    if method=="FREAK":
        descsA=descsA[1]
        descsB=descsB[1]
        
    # create a BFMatcher object which will match up the SIFT features
    # select measurement for the BFMatcher  
    distance_type = define_measurement(method)
    

    # Useful info about DMatch objects -> https://docs.opencv.org/java/2.4.9/org/opencv/features2d/DMatch.html
    
    # Declare objects
    if method == "SURF" or method == "HOG":
        bf = cv2.BFMatcher(distance_type)
    else:
        bf = cv2.BFMatcher(distance_type, crossCheck=True)
            
    # Generate matches   
    if method == "HOG":
        match = bf.knnMatch(descsA, descsB, 1)
        matches = [item for sublist in match for item in sublist]
        
    else:
        
        matches = bf.match(descsA, descsB)
    
    # Sort the matches in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # keep N top matches
    matches = matches[0:N]
    if(plot == True):
        # Plots both images + theirs coincident matches
        plot_matches(siftA, siftB, pathA, pathB, matches, resize)
        
    return matches


def retreive_image(siftDs, siftQueries, paths, k, th = 60, descsMin = 3, method="SIFT", plot = False, resize = False):  
    queriesResult = []
    distancesResult = []
    finalMatch=[]
    i = 0
    l = len(siftQueries)
    for imNameQuery in siftQueries:
        matches = []
        siftQuery = siftQueries[imNameQuery]
        print('Query', str(i+1)+'/'+str(l),'started.')
        i+=1    
        for imNameDs in siftDs:
            siftIm = siftDs[imNameDs]
            
            # As a default just return 100 best matches per image, could be incresed
            matchesBF = BFMatcher(100, siftQuery, siftIm, method, pathA=paths['pathQueriesValidation'], 
                                  pathB = paths['pathDS'], plot = plot, resize = resize)
            
            
            distance = [o.distance for o in matchesBF if o.distance <= th]
            
            matches.append([imNameDs, distance])

        # Sort images per number of matches under threshold level
        matches = sorted(matches, key = lambda x:len(x[1]), reverse = True)
        
        if(len(matches) > k):
            matches = matches[0:k]
        
        # Detect if image is not present in the Dataset
        tots=0
        for index,row in enumerate(matches):
            # Comprobar si tots son mes petits a un threshold
            if len(row[1]) < descsMin:
               tots+=1
              
        # Contruct query result to be returend
        distancesResult.append([ row[1] for row in matches ])
        queriesResult.append([ row[0]  for row in matches ] if tots<10 else [-1])
        
        finalMatch.append(matches)
    
    
    return queriesResult, distancesResult, finalMatch

# Computes distances taking into account GT pairs
def get_gt_distance(N, sift_ds, sift_validation, gt_list, paths, method,resize = False):
    validationMatches = []
    
    for i,imName in enumerate(sift_validation):
        imQuery = gt_list[i][0]
        
        print("\n Query: ", imQuery)
        if(imQuery == -1):    
            # Image not in the DS
            pass
        
        else:
            siftA = sift_validation[imName]
            siftB = sift_ds[imQuery]
            matchesBF = BFMatcher(N, siftA, siftB, method,pathA=paths['pathQueriesValidation'], 
                                  pathB = paths['pathDS'], plot = True, resize = resize)  
            
                        
            validationMatches.append([imName, [o.distance for o in matchesBF] ])
    
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

def remove_kps(siftDict, area):
# area dictionary of list => [tlx, tly, brx, bry]
# key same as for SiftDicts, image names.
    for entry in siftDict:
        name, kps, descs = siftDict[entry]
        
        if len(area[name]) >0:    
            tlx, tly, brx, bry = area[name]
            i = len(kps)
            for kp in reversed(kps):
                kpx, kpy = kp.pt
                if(kpy < bry and kpy > tly):
                    if(kpx < brx and kpx > tlx):
                    # KPs withing forgiben area
                        kps.pop(i-1)
                        descs = np.delete(descs,i-1,0)   
                i -= 1
            siftDict[entry] = [name, kps, descs]
    return siftDict