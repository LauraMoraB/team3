import random
from utils import save_pkl, mapk, create_dir, plot_gray, plot_rgb, plot_sift, plot_matches
from sift import compute_sift, BFMatcher


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
    sift_ds = compute_sift(paths['pathDS'])
    sift_validation = compute_sift(paths['pathQueriesValidation'])
    # Demmonstration of status
    demo()
