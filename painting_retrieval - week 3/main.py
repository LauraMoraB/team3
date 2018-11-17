from utils import save_pkl, mapk, create_dir, get_query_gt, slice_dict, plot_sift
from sift import compute_sift, get_gt_distance, get_distances_stats, retreive_image, compute_threshold
import time
from flann import retreive_image_withFlann
from argparse import ArgumentParser
from detectText import detect_text_bbox

import configparser

def init(mode):
    # --> BEGINING FOLDERS PREPARATION <-- #
    paths = {}
    # Images Path --> Add new entry in the dictionary for new subfodlers!
    paths['pathDS'] = "dataset/"
    paths['pathQueriesValidation'] = "queries_validation/"
    paths['pathGTValidation'] = "queries_validation/GT/"
    paths['pathQueriesTest'] = "queries_test/"
    paths['pathGTTest'] = "queries_test/GT/"
    # Results Path
    paths['pathResult'] = "results/"+mode
    
    # Delivery Methods Path
    paths['pathResults1'] = paths['pathResult']+"/sift/"
    paths['pathResults2'] = paths['pathResult']+"/rootsift/"
    paths['pathResults3'] = paths['pathResult']+"/orb/"
    paths['pathResults4'] = paths['pathResult']+"/kaze/"
    paths['pathResults5'] = paths['pathResult']+"/surf/"

    
    # Create all subdirectories on dictionary if they dont already
    for path in paths:
        create_dir(paths[path])
    print('All subfolders have been created')
    
    # --> END FOLDERS PREPARATION <-- #
    return paths
    
if __name__ == "__main__":
    
    global CONSOLE_ARGUMENTS
    
    def parse_arguments():
        """
    	Parse line arguments
    	"""
        
        parser = ArgumentParser()
        
        general_args = parser.add_argument_group("General arguments")
    
        
        general_args.add_argument('-me', '--method', default="SIFT", choices=('SIFT', 'ORB', 'KAZE', 'SURF','HOG'))
        general_args.add_argument('-ma', '--matcher',  default="BFMatcher",choices=('BFMatcher', 'Flann'))
        general_args.add_argument("-t", "--text", default=True, action='store_true', help="Detect text")
        general_args.add_argument("-rs", "--rootsift", default=True, action='store_true', help="Only for sift method")

        # create our group of mutually exclusive arguments
        mutually_exclusive = parser.add_mutually_exclusive_group()
        mutually_exclusive.add_argument("--test", action='store_true', help="test excludes validate")
        mutually_exclusive.add_argument("--validate", action='store_true', help="validate excludes test")
        
        
        return parser.parse_args()


    # init arguments
    CONSOLE_ARGUMENTS = parse_arguments()
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    
    RELOAD = config.getboolean('DEFAULT','RELOAD')
    GT_MATCHING = config.getboolean('DEFAULT','GT_MATCHING')
    RETRIEVAL = config.getboolean('DEFAULT','RETRIEVAL')
    SAVE_RESULTS = config.getboolean('DEFAULT','SAVE_RESULTS')
    RESIZE = config.getboolean('DEFAULT','RESIZE')
    PLOTS = config.getboolean('DEFAULT','PLOTS')

    
    QUERY_SET_TRAIN=CONSOLE_ARGUMENTS.validate  
    QUERY_SET_TEST=CONSOLE_ARGUMENTS.test
    
    if QUERY_SET_TRAIN == True:
        MODE = "test"
    else:
        MODE = "validation"
        

    
    # SET OPTION IN THE DEFAULT VALUE IN parse_arguments
    method = CONSOLE_ARGUMENTS.method
    ROOTSIFT = CONSOLE_ARGUMENTS.rootsift
    matcherType = CONSOLE_ARGUMENTS.matcher
    
    TEXT = CONSOLE_ARGUMENTS.text
    
    if(RELOAD):
        # Prepares folders
        paths = init(MODE)

        gtFile = "queries_validation/GT/w5_query_devel.pkl"
        gtList = get_query_gt(gtFile)
        
        # Creates dictionary of list with SIFT kps and descriptors  
        # FORMAT-> sift['imName']= [imName, kps, descs]
        
        print ("Computing Features and Descriptors for dataset..")
        start = time.time()
        
        siftDs = compute_sift(paths['pathDS'], method, resize = RESIZE, rootSift = ROOTSIFT)
        
        end = time.time()
        tTime= end - start
        print('Total time:',tTime)
        
        if MODE == "validation":
            path = paths['pathQueriesValidation']
        else:
            path = paths['pathQueriesTest']
            
        siftQuery = compute_sift(path, method, resize = RESIZE, rootSift = ROOTSIFT)

    
    if (TEXT):
        
        list_of_text_bbox = detect_text_bbox(paths['pathDS'], plot=True)
    
        # save pkl
        save_pkl(list_of_text_bbox, "TextResults/")
        
    
    if(GT_MATCHING):
        
        # N Used for Stats  and plotting
        N = 20
        # Matches Validation query with their GT correspondences
        gtMatches = get_gt_distance(N, siftDs, siftQuery, gtList, paths, 
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
                                    siftQuery, paths, k, method,th, descsMin)
            
        elif matcherType == "BFMatcher":
            queriesResult, distancesResult, matches=retreive_image(siftDs, 
                                    siftQuery, paths, k, th, descsMin,
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
        
        elif method=="KAZE":
            pathResult =  paths['pathResults4']
        
        elif method=="SURF":
            pathResult =  paths['pathResults5']
        
        
        save_pkl(queriesResult, pathResult)


        
