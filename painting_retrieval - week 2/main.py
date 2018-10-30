import random
import numpy as np
from utils import save_pkl, mapk, create_dir, plot_gray, plot_rgb, plot_sift, plot_matches, get_query_gt
from sift import compute_sift, BFMatcher, get_gt_distance, get_distances_stats

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
    siftA = sift_ds[random.choice(list(sift_ds.keys()))]
    plot_sift(siftA, paths['pathDS'])
    print('Sift matching example on random image from ds:')
    siftA = sift_ds[random.choice(list(sift_ds.keys()))]
    siftB = sift_ds[random.choice(list(sift_ds.keys()))]
    BFMatcher(50, siftA, siftB, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True)   
    
if __name__ == "__main__":

    RELOAD = False
    if(RELOAD):
        # Prepares folders
        paths = init()
        # Loads GT
        GT_file = "queries_validation/GT/query_corresp_simple_devel.pkl"
        gt_list = get_query_gt(GT_file)
        # Creates list of list with sift kps and descriptors  -> [imName, kps, descs]
        sift_ds = compute_sift(paths['pathDS'])
        sift_validation = compute_sift(paths['pathQueriesValidation'])

    GT_MATCHING = False
    if(GT_MATCHING):
        N = 25
        gt_matches = get_gt_distance(N, sift_ds, sift_validation, gt_list, paths)
        gt_stats = get_distances_stats(N, gt_matches, plot = True)
        
                
        
