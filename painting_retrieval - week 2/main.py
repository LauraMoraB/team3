import cv2
from utils import list_ds, save_pkl, mapk, create_dir, get_query_gt, plot_gray, get_gray_image

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
    # --> END FOLDERS PREPARATION <-- #
    return paths

def load_sift_descriptors(path, method = 0):
    
    im_list = list_ds(path)
    sift_result = []
    # Creates SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    for imName in im_list:
        # Load Gray version of each image
        imGray = get_gray_image(imName, path)
        # Find KeyLpoints
        (kps, descs) = sift.detectAndCompute(imGray, None)
        sift_result.append([imName, kps, descs])    
        print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    return sift_result
        
        


#
#if pass_queries == True:
#    X2resultList = []
#    HIresultList = []
#    HKresultList = []
#    # Create list of lists for all histograms in the query test/evaluation
#    if(performEvaluation == 1):
#        if(method == 1 or method == 2):
#            for index, row in dfQuery.iterrows():
#                queryImage = row["Image"]
#                imgBGR = cv2.imread(pathQueries+queryImage,1)
#                if prepoces==True:
#                    global_color(imgBGR, spaceType, pathprep_resultQueries, queryImage, True)    
#                else:
#                    global_color(imgBGR, spaceType, pathprep_resultQueries, queryImage, False)
#
#            store_histogram_total(dfQuery, pathprep_resultQueries+"Final/", spaceType, level=level)	
#            whole_query_list = [histograms_to_list(row_ds, level, spaceType) for _,row_ds in dfQuery.iterrows() ]
#        elif(method ==3):
#            whole_query_list = texture_method1(dfQuery, pathQueries)       
#    
#    elif(performTest == 1):
#        if(method == 1 or method == 2):
#            for index, row in dfQueryTest.iterrows():
#                queryImage = row["Image"]
#                imgBGR = cv2.imread(pathQueriesTest+queryImage,1)
#                if prepoces==True:
#                    global_color(imgBGR, spaceType, pathprep_resultQueries, queryImage, True)    
#                else:
#                    global_color(imgBGR, spaceType, pathprep_resultQueries, queryImage, False)
#
#            store_histogram_total(dfQueryTest, pathprep_resultQueries+"Final/", spaceType, level=level)	
#            whole_query_list = [histograms_to_list(row_ds, level, spaceType) for _,row_ds in dfQueryTest.iterrows() ]
#        elif(method ==3):
#            whole_query_list = texture_method1(dfQueryTest, pathQueriesTest)
#            
#    # Create list of lists for all histograms in the dataset
#    if(method == 1 or method == 2):         
#        whole_hist_list = [histograms_to_list(row_ds, level, spaceType) for _,row_ds in dfDataset.iterrows() ]
#    elif(method ==3):
#        whole_hist_list = texture_method1(dfDataset, pathDS)
#
#    for query in whole_query_list:
#        histogram_query = query
#
#        X2resultList.append(getX2results(whole_hist_list, histogram_query,  k, dfDataset))
#        HIresultList.append(getHistInterseccionResult(whole_hist_list, histogram_query,  k, dfDataset))
#        HKresultList.append(getHellingerKernelResult(whole_hist_list, histogram_query,  k, dfDataset))
#
#    if(performEvaluation == 1):
#        # Load provided GT
#        actualResult = get_query_gt(GT_file)
#        # Validation -> MAPK RESULT
#        mapkX2 = mapk(actualResult, X2resultList, k)
#        print('MAPK score using X2:',mapkX2)
#        mapkKI = mapk(actualResult, HIresultList, k)
#        print('MAPK score using HI:',mapkKI)
#        mapkHK = mapk(actualResult, HKresultList, k)
#        print('MAPK score using HK:',mapkHK)
#
#    elif(performTest == 1):
#    # Save results for X2 whicih gives best performance
#        if(method == 1):
#            save_pkl(X2resultList, pathResultsM1)
#        elif(method == 2):
#            save_pkl(X2resultList, pathResultsM2)
#        elif(method == 3):
#            save_pkl(X2resultList, pathResultsM3)
#

if __name__ == "__main__":

    paths = init()
    sift = load_sift_descriptors(paths['pathDS'])
    
