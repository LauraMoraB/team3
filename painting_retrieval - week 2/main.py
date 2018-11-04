import random
from utils import save_pkl, mapk, create_dir, get_query_gt, slice_dict, plot_sift
from sift import compute_sift, BFMatcher, get_gt_distance, get_distances_stats, retreive_image, compute_threshold
import time
from flann import retreive_image_withFlann

def init():
    # --> BEGINING FOLDERS PREPARATION <-- #
    paths = {}
    # Images Path --> Add new entry in the dictionary for new subfodlers!
    paths['pathDS'] = "dataset/"
    paths['pathQueriesValidation'] = "queries_validation/"
    paths['pathGTValidation'] = "queries_validation/GT/"
    paths['pathQueriesTest'] = "queries_test/"
    paths['pathGTTest'] = "queries_test/GT/"
    # Results Path
    paths['pathResult'] = "results/"
    
    # Delivery Methods Path
    paths['pathResults1'] = "results/sift/"
    paths['pathResults2'] = "results/rootsift/"
    paths['pathResults3'] = "results/orb/"
    
    # Create all subdirectories on dictionary if they dont already
    for path in paths:
        create_dir(paths[path])
    print('All subfolders have been created')
    
    # --> END FOLDERS PREPARATION <-- #
    return paths

def demo():
    # Example for ploting a sift image
    print('Sift kps example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    plot_sift(siftA, paths['pathDS'], resize = False)
    print('Sift matching example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    siftB = siftDs[random.choice(list(siftDs.keys()))]
    method = "DAISY"
    BFMatcher(50, siftA, siftB, method, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True)   
    
if __name__ == "__main__":
    
    RELOAD = True
    GT_MATCHING = False
    RETRIEVAL = True
    ROOTSIFT = False
    SAVE_RESULTS = False
    RESIZE = True
        PLOTS = False
    
    # Define which Descriptor is used
    # OPTIONS: SIFT/ ORB / DAISY / KAZE / FREAK / HOG
    # IF ORB IS SELECTED, ROOTSIFT = FALSE
    method = "HOG"
    # OPTIONS BFMatcher / Flann
    matcherType = "BFMatcher"
    
    if(RELOAD):
        # Prepares folders
        paths = init()
        # Loads GT (from previous week, ds not available at the moment)
        gtFile = "queries_validation/GT/w4_query_devel.pkl"
        gtList = get_query_gt(gtFile)
        # Creates dictionary of list with SIFT kps and descriptors  
        # FORMAT-> sift['imName']= [imName, kps, descs]
        print ("Computing Features and Descriptors for dataset..")
        
        start = time.time()
        siftDs = compute_sift(paths['pathDS'], method, resize = RESIZE, rootSift = ROOTSIFT)
        end = time.time()
        tTime= end - start
        print('Total time:',tTime)
        
        siftValidation = compute_sift(paths['pathQueriesValidation'], method, resize = RESIZE, rootSift = ROOTSIFT)

    if(GT_MATCHING):
        
        # N Used for Stats  and plotting
        N = 20
        # Matches Validation query with their GT correspondences
        gtMatches = get_gt_distance(N, siftDs, siftValidation, gtList, paths, 
                                    method,
                                    resize = RESIZE)
        # Compute distance Stats for GT correspondences
        gtStats = get_distances_stats(N, gtMatches, plot = PLOTS)


    if(RETRIEVAL):   
        # Number of images retrieval per query
        k = 10
        # Max distance to consider good matches
        
        th, descsMin = compute_threshold(matcherType, method, ROOTSIFT)
        
        
        # Min number of matches to considerer a good retrieval
        # Returns queries retrival + their distances + debugging & tuning
        start = time.time()
        print("Starting comparison...")
        
        if matcherType == "Flann":
            queriesResult, distancesResult, matches = retreive_image_withFlann(siftDs, 
                                    siftValidation, paths, k, method,th, descsMin)
            
        elif matcherType == "BFMatcher":
            queriesResult, distancesResult, matches=retreive_image(siftDs, 
                                    siftValidation, paths, k, th, descsMin,
                                    method, PLOTS, RESIZE)
        else:
            print ("Invalid Matcher")
        
        
        end = time.time()
        tTime= end - start
        print('Total time:',tTime)
        
        # Evaluation
        for n in range(k):
            mapkResult = mapk(gtList, queriesResult, n+1)
            print('MAPK@'+str(n+1)+':',mapkResult)
            
    # Save Results, modify path accordingly to the  Method beeing used
    if(SAVE_RESULTS):
        if method == "SIFT":
            if(ROOTSIFT == False):
                pathResult =  paths['pathResults1']
            else:
                pathResult =  paths['pathResults2']
        elif method == "ORB":
            pathResult =  paths['pathResults3']
            
        save_pkl(queriesResult, pathResult)


        
