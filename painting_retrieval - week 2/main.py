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
        # Loads GT
        gtFile = "queries_validation/GT/query_corresp_simple_devel.pkl"
        gtList = get_query_gt(gtFile)
        # Creates list of list with sift kps and descriptors  -> [imName, kps, descs]
        siftDs = compute_sift(paths['pathDS'])
        siftValidation = compute_sift(paths['pathQueriesValidation'])

    GT_MATCHING = False
    if(GT_MATCHING):
        N = 30
        gtMatches = get_gt_distance(N, siftDs, slice_dict(siftValidation,0,1), gtList, paths)
        gtStats = get_distances_stats(N, gtMatches, plot = True)
    
    k = 10
    quesriesResult, distancesResult = retreive_image(siftDs, siftValidation, paths, k, th = 100)
    # Evaluation
    mapk(gtList, quesriesResult)

        
