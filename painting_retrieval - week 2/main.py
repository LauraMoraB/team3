import random
import numpy as np
from utils import save_pkl, mapk, create_dir, plot_gray, plot_rgb, plot_sift, plot_matches, get_query_gt, slice_dict
from sift import compute_sift, BFMatcher, get_gt_distance, get_distances_stats, retreive_image

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

    RELOAD = False
    if(RELOAD):
        # Prepares folders
        paths = init()
        # Loads GT (from previous week, ds not available at the moment)
        gtFile = "queries_validation/GT/query_corresp_simple_devel.pkl"
        gtList = get_query_gt(gtFile)
        # Creates dictionary of list with SIFT kps and descriptors  
        # FORMAT-> sift['imName']= [imName, kps, descs]
        siftDs = compute_sift(paths['pathDS'])
        siftValidation = compute_sift(paths['pathQueriesValidation'])

    GT_MATCHING = False
    if(GT_MATCHING):
        # N Used for Stats polotting
        N = 30
        # Matches Validation query with their GT correspondences
        gtMatches = get_gt_distance(N, siftDs, siftValidation, gtList, paths)
        # Compute distance Stats for GT correspondences
        gtStats = get_distances_stats(N, gtMatches, plot = True)

    # --> BEGINING Image Retrieval Sift + BF <-- #
    
    # Number of images retrieval per query
    k = 10
    # Max distance to consider a good match
    th = 100
    # Min number of matches to considerer a good retrieval
    descsMin = 3
    
    quesriesResult, distancesResult = retreive_image(siftDs, 
                                                     slice_dict(siftValidation,0,1), 
                                                     paths, k, th, descsMin)
    # Evaluation
    mapkResult = mapk(gtList, quesriesResult, k)
    print('MAPK@'+str(k)+':',mapkResult)

        
