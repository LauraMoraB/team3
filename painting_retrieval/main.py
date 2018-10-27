import cv2
from utils import create_df, get_full_image, save_pkl, mapk, plot_rgb, plot_gray, create_dir, get_query_gt
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

# Crear directorios
create_dir(pathResults)
create_dir(pathprep_resultDS)
create_dir(pathprep_resultQueries)


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


dfDataset = create_df(pathDS)
dfQuery = create_df(pathQueries)

if global_color_histograms==True:
	for i in range(len(dfDataset)):
		dfSingle = dfDataset.iloc[i]
		imgBGR = get_full_image(dfSingle, pathDS)
		imageName = dfSingle['Image']
		channel0Single, channel1Single, channel2Single = global_color_hist(imgBGR, spaceType, pathprep_resultDS, imageName)
		save_global_color_hist(channel0Single, channel1Single, channel2Single, dfSingle,spaceType, imageName,pathResults)


if build_dataset==True:
    # Read Images
    if prepoces==True:
        for index, row in dfDataset.iterrows():
            imageName = row["Image"]
            imgBGR = cv2.imread(pathDS+imageName,1)

            global_color(imgBGR, spaceType, pathprep_resultDS, imageName)

        store_histogram_total(dfDataset, pathprep_resultDS+"equalyse_luminance/", spaceType, level=level)

    else:
        # Save image descriptors
        store_histogram_total(dfDataset, pathDS, spaceType, level=level)


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
    whole_hist_list = [histograms_to_list(row_ds, level, spaceType) for _,row_ds in dfDataset.iterrows() ]

    for index,row in dfQuery.iterrows():
        histogram_query = histograms_to_list(row, level, spaceType)
        queryList.append(row['Image'])

        X2resultList.append(getX2results(whole_hist_list, histogram_query,  k, dfDataset))
        HIresultList.append(getHistInterseccionResult(whole_hist_list, histogram_query,  k, dfDataset))
        HKresultList.append(getHellingerKernelResult(whole_hist_list, histogram_query,  k, dfDataset))


    # Load provided GT
    actualResult = get_query_gt(GT_file)
    # Validation -> MAPK RESULT
    mapkX2 = mapk(actualResult, X2resultList, k)
    print('MAPK score using X2:',mapkX2)
    mapkKI = mapk(actualResult, HIresultList, k)
    print('MAPK score using HI:',mapkX2)
    mapkHK = mapk(actualResult, HKresultList, k)
    print('MAPK score using HK:',mapkX2)

    # Save results into pkl format
    save_pkl(X2resultList, pathResults)
    save_pkl(HIresultList, pathResults)
    save_pkl(HKresultList, pathResults)
