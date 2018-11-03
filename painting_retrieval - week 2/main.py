import random
from utils import save_pkl, mapk, create_dir, get_query_gt, slice_dict, plot_sift
from sift import compute_sift, BFMatcher, get_gt_distance, get_distances_stats, retreive_image

from Surf import compute_suft, get_gt_distance_nearest, Matcher_find_nearest,retreive_image_nearest
import time

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
    
    # Create all subdirectories on dictionary if tey dont already
    for path in paths:
        create_dir(paths[path])
    print('All subfolders have been created')
    # --> END FOLDERS PREPARATION <-- #
    return paths

def demosift():
    # Example for ploting a sift image
    print('Sift kps example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    plot_sift(siftA, paths['pathDS'], resize = False)
    print('Sift matching example on random image from ds:')
    siftA = siftDs[random.choice(list(siftDs.keys()))]
    siftB = siftDs[random.choice(list(siftDs.keys()))]
    BFMatcher(50, siftA, siftB, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True)   
    
def demosurf():
    # Example for ploting a sift image
    print('Surf kps example on random image from ds:')
    surfA = surftDs[random.choice(list(surftDs.keys()))]
    plot_sift(surfA, paths['pathDS'], resize = False)
    print('Surf matching example on random image from ds:')
    surfA = surftDs[random.choice(list(surftDs.keys()))]
    surfB = surftDs[random.choice(list(surftDs.keys()))]
    Matcher_find_nearest(50, surfA, surfB, pathA = paths['pathDS'], pathB = paths['pathDS'], plot = True) 
    
if __name__ == "__main__":

    RELOAD = True
    GT_MATCHING = True
    RETRIEVAL = True
    ROOTSIFT = False
    SAVE_RESULTS = False
    RESIZE = False
    PLOTS = False
    metodo='SURF' #'SIFT' #'SURF'
    #demosurf()
    if(RELOAD):
        # Prepares folders
        paths = init()
        # Loads GT (from previous week, ds not available at the moment)
        gtFile = "queries_validation/GT/w4_query_devel.pkl"
        gtList = get_query_gt(gtFile)
        # Creates dictionary of list with SIFT kps and descriptors  
        # FORMAT-> sift['imName']= [imName, kps, descs]
        if (metodo=='SIFT'):
            siftDs = compute_sift(paths['pathDS'], resize = RESIZE, rootSift = ROOTSIFT)
            siftValidation = compute_sift(paths['pathQueriesValidation'], resize = RESIZE, rootSift = ROOTSIFT)
        else:
            surftDs = compute_suft(paths['pathDS'])
            surtValidation = compute_suft(paths['pathQueriesValidation'])

    if(GT_MATCHING):
        # N Used for Stats  and plotting
        N = 20
        if (metodo=='SIFT'):
            # Matches Validation query with their GT correspondences
            gtMatches = get_gt_distance(N, siftDs, siftValidation, gtList, paths, resize = RESIZE)
            # Compute distance Stats for GT correspondences
            gtStats = get_distances_stats(N, gtMatches, plot = PLOTS)
            
        else:
            
            gtMatches = get_gt_distance_nearest(N, surftDs, surtValidation, gtList, paths, resize = RESIZE)
            gtStats = get_distances_stats(N, gtMatches, plot = PLOTS)


            

    if(RETRIEVAL):   
        # Number of images retrieval per query
        k = 10
        # Max distance to consider good matches
        if(ROOTSIFT == False):
            th = 90
            descsMin = 15
        else:
            th = 0.15
            descsMin = 5
        # Min number of matches to considerer a good retrieval
        # Returns queries retrival + theis distances + debugging & tuning
        start = time.time()
        if (metodo=='SIFT'):
            queriesResult, distancesResult = retreive_image(siftDs, 
                                                         siftValidation,#slice_dict(siftValidation,29,30), 
                                                         paths, k, th, descsMin, PLOTS, RESIZE)
        else:
            ratio_thresh = 0.7
            queriesResult, distancesResult = retreive_image_nearest(surftDs, 
                                                         surtValidation,#slice_dict(siftValidation,29,30), 
                                                         paths, k, ratio_thresh , descsMin, PLOTS, RESIZE)
           
        end = time.time()
        tTime= end - start
        print('Total time:',tTime)
        
        # Evaluation
        for n in range(k):
            mapkResult = mapk(gtList, queriesResult, n+1)
            print('MAPK@'+str(n+1)+':',mapkResult)
            
    # Save Results, modify path accordingly to the  Method beeing used
    if(SAVE_RESULTS):   
        if(ROOTSIFT == False):
            pathResult =  paths['pathResults1']
        else:
            pathResult =  paths['pathResults2']
        save_pkl(queriesResult, pathResult)


        
