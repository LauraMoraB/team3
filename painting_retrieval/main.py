import cv2
from utils import create_df, get_full_image, save_pkl, mapk, create_dir, get_query_gt
from method1 import store_histogram_total, histograms_to_list
from task5 import texture_method1
from global_color_histograms import global_color_hist,save_global_color_hist, global_color
from task3 import getHellingerKernelResult, getHistInterseccionResult, getX2results

# Paths
pathDS = "dataset/"
pathQueries = "queries/"
pathResults = "results/"
pathprep_resultDS = "results_preprocesadoDS/"
pathprep_resultQueries = "results_preprocesadoQueries/"
GT_file = "queries/GT/query_corresp_simple_devel.pkl"
pathResultsM1 = "results/method1/"
pathResultsM2 = "results/method2/"
pathResultsM3 = "results/method3/"
pathQueriesTest = "queries_test/"

# Crear directorios
create_dir(pathprep_resultDS)
create_dir(pathprep_resultQueries)
create_dir(pathResultsM1)
create_dir(pathResultsM2)
create_dir(pathResultsM3)

# Number of results per query
k = 10
#Build database
build_dataset=True
#Make queries
pass_queries=True
#choose prepoces
prepoces = False
#choose global_color_histograms: image will be procesed and change space color and save global_color_hist in resuts_GVHistogram (create file )
global_color_histograms = False
#Numer of partitions of histograms
level=0
#type of space
spaceType= "HSV" #"BGR" #"HSV", "HSL","LAB", "YCrCb","XYZ","LUV"
# Final evaluation and Test
performEvaluation = 1
performTest = 0
# which of the three different final methods is performed, 1 BGR, 2 LUV, 3 Wavelet
method = 3


dfDataset = create_df(pathDS)
dfQuery = create_df(pathQueries)
dfQueryTest = create_df(pathQueriesTest)

if global_color_histograms==True:
	for i in range(len(dfDataset)):
		dfSingle = dfDataset.iloc[i]
		imgBGR = get_full_image(dfSingle, pathDS)
		imageName = dfSingle['Image']
		channel0Single, channel1Single, channel2Single = global_color_hist(imgBGR, spaceType, pathprep_resultDS, imageName)
		save_global_color_hist(channel0Single, channel1Single, channel2Single, dfSingle,spaceType, imageName,pathResults)


if build_dataset==True:
    # Read Images
    for index, row in dfDataset.iterrows():
        imageName = row["Image"]
        imgBGR = cv2.imread(pathDS+imageName,1)
        if prepoces==True:
            global_color(imgBGR, spaceType, pathprep_resultDS, imageName, True)
        else:
            global_color(imgBGR, spaceType, pathprep_resultDS, imageName, False)

    store_histogram_total(dfDataset, pathprep_resultDS+"Final/", spaceType, level=level)


if pass_queries == True:
    X2resultList = []
    HIresultList = []
    HKresultList = []

    queryList = []

    if prepoces ==True :
        for index, row in dfQuery.iterrows():
            queryImage = row["Image"]
            imgBGR = cv2.imread(pathQueries+queryImage,1)

            global_color(imgBGR, spaceType, pathprep_resultQueries, queryImage)

        store_histogram_total(dfQuery,pathprep_resultQueries+"equalyse_luminance/", spaceType, level=level)
    else:
        store_histogram_total(dfQuery,pathQueries, spaceType, level=level)


    # Create list of lists for all histograms in the dataset
    if(method == 1 or method == 2):         
        whole_hist_list = [histograms_to_list(row_ds, level, spaceType) for _,row_ds in dfDataset.iterrows() ]
    elif(method ==3):
        whole_hist_list = texture_method1(dfDataset, pathDS)
        
    if(performEvaluation == 1):
        if(method == 1 or method == 2):         
            whole_query_list = texture_method1(dfQuery, pathQueries)
        elif(method ==3):
            whole_query_list = texture_method1(dfQuery, pathQueries)            
    elif(performTest == 1):
        if(method == 1 or method == 2):         
            whole_query_list = texture_method1(dfQueryTest, pathQueriesTest)
        elif(method ==3):
            whole_query_list = texture_method1(dfQueryTest, pathQueriesTest)

    for query in whole_query_list:
        histogram_query = query

        X2resultList.append(getX2results(whole_hist_list, histogram_query,  k, dfDataset))
        HIresultList.append(getHistInterseccionResult(whole_hist_list, histogram_query,  k, dfDataset))
        HKresultList.append(getHellingerKernelResult(whole_hist_list, histogram_query,  k, dfDataset))

    if(performEvaluation == 1):
        # Load provided GT
        actualResult = get_query_gt(GT_file)
        # Validation -> MAPK RESULT
        mapkX2 = mapk(actualResult, X2resultList, k)
        print('MAPK score using X2:',mapkX2)
        mapkKI = mapk(actualResult, HIresultList, k)
        print('MAPK score using HI:',mapkKI)
        mapkHK = mapk(actualResult, HKresultList, k)
        print('MAPK score using HK:',mapkHK)

    elif(performTest == 1):
    # Save results for X2 whicih gives best performance
        if(method == 1):
            save_pkl(X2resultList, pathResultsM1)
        elif(method == 2):
            save_pkl(X2resultList, pathResultsM2)
        elif(method == 3):
            save_pkl(X2resultList, pathResultsM3)