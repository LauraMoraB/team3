import random
import numpy as np
import cv2
from utils import save_pkl, mapk, create_dir, get_query_gt, slice_dict, plot_sift,list_ds
from sift import compute_sift, BFMatcher, get_gt_distance, get_distances_stats, retreive_image

def init():
    # --> BEGINING FOLDERS PREPARATION <-- #
    paths = {}
    # Images Path --> Add new entry in the dictionary for new subfodlers!
#    paths['pathDS'] = "dataset/"
#    paths['pathQueriesValidation'] = "queries/"

    paths['pathDS'] = "BBDD_W4/"
    paths['pathQueriesValidation'] = "query_devel_W4/"
#    paths['pathQueriesGiro'] = "Giro/"
    paths['pathGTValidation'] = "queries_validation/GT/"
    paths['pathQueriesTest'] = "queries_test/"
    paths['pathGTTest'] = "queries_test/GT/"
    # Results Path
    paths['pathResult'] = "results/"
    # Delivery Methods Path
    paths['pathResults1'] = "results/sift/"
    paths['pathResults2'] = "results/rootsift/"
    
    # Create all subdirectories on dictionary if tey dont already
    for path in paths:
        create_dir(paths[path])
    print('All subfolders have been created')
    # --> END FOLDERS PREPARATION <-- #
    return paths

def demo():
    # Example for ploting a sift image
    print('Sift kps example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    plot_sift(siftA, paths['pathDS'])
    print('Sift matching example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    siftB = siftDs[random.choice(list(siftDs.keys()))]
    BFMatcher(50, siftA, siftB, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True)   
    
if __name__ == "__main__":

    RELOAD = True
    GT_MATCHING = True
    RETRIEVAL = True
    ROOTSIFT = True
    SAVE_RESULTS = True
    
#    paths = init()
#    im_list = list_ds(paths['pathQueriesValidation'] )
#
#    for imName in im_list:
#        # Load Gray version of each image
#        imBGR = cv2.imread(paths['pathQueriesValidation'] +imName)
#    
#        rows,cols,ch = imBGR.shape
#    
#        pts1 = np.float32([[50,50],[200,50],[50,200]])
#        pts2 = np.float32([[10,100],[200,50],[100,250]])
#        
#        M = cv2.getAffineTransform(pts1,pts2)
#        
#        dst = cv2.warpAffine(imBGR,M,(cols,rows))
#        cv2.imwrite(paths['pathQueriesGiro']+imName, dst)

    if(RELOAD):
        # Prepares folders
        paths = init()
        # Loads GT (from previous week, ds not available at the moment)
        gtFile = "queries_validation/GT/query_corresp_simple_devel.pkl"
        gtList = get_query_gt(gtFile)
        # Creates dictionary of list with SIFT kps and descriptors  
        # FORMAT-> sift['imName']= [imName, kps, descs]
        siftDs = compute_sift(paths['pathDS'], rootSift = ROOTSIFT, eps=1e-7)
        


        siftValidation = compute_sift(paths['pathQueriesValidation'], rootSift = ROOTSIFT, eps=1e-7)

    if(GT_MATCHING):
        # N Used for Stats  and plotting
        N = 20
        # Matches Validation query with their GT correspondences
        gtMatches = get_gt_distance(N, siftDs, siftValidation, gtList, paths)
        # Compute distance Stats for GT correspondences
        gtStats = get_distances_stats(N, gtMatches, plot = True)

    if(RETRIEVAL):   
        # Number of images retrieval per query
        k = 10
        # Max distance to consider good matches
        if(ROOTSIFT == False):
            th = 100
        else:
            th = 0.2
        # Min number of matches to considerer a good retrieval
        descsMin = 2
        # Returns queries retrival + theis distances + debugging & tuning
        quesriesResult, distancesResult = retreive_image(siftDs, 
                                                         siftValidation,#slice_dict(siftValidation,0,1), 
                                                         paths, k, th, descsMin)
        # Evaluation
        for n in range(k):
            mapkResult = mapk(gtList, quesriesResult, n+1)
            print('MAPK@'+str(n+1)+':',mapkResult)
            
    # Save Results, modify path accordingly to the  Method beeing used
    if(SAVE_RESULTS):   
        if(ROOTSIFT == False):
            pathResult =  paths['pathResults1']
        else:
            pathResult =  paths['pathResults2']
        save_pkl(quesriesResult, pathResult)


        
