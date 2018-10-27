import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import create_df, get_full_image, submission_list, save_pkl, mapk, get_image, plot_rgb, plot_gray, create_dir
from method1 import store_histogram_total, histograms_to_list
from task5 import haar_wavelet, haar_sticking
from task3 import getX2results
from global_color_histograms import global_color_hist,save_global_color_hist, global_color
import pandas as pd

# Paths
pathDS = "dataset/"
pathQueries = "queries/query_devel_random/"
pathResults = "results/"
pathprep_resultDS = "results_preprocesadoDS/"
pathprep_resultQueries = "results_preprocesadoQueries/"

create_dir(pathResults)
create_dir(pathprep_resultDS)
create_dir(pathprep_resultQueries)


# Number of results per query
k = 10
build_dataset=True
pass_queries=False
level=0

#type of space 
spaceType= "HSV" #"HSV", "HSL","LAB", "YCrCb","XYZ","LUV"

#choose prepoces
prepoces =True


#choose global_color_histograms: image will be procesed and change space color and save global_color_hist in resuts_GVHistogram (create file )
global_color_histograms = False

if global_color_histograms==True:
	for i in range(len(dfDataset)):       
		dfSingle = dfDataset.iloc[i]
		imgBGR = get_full_image(dfSingle, pathDS)    
		imageName = dfSingle['Image']  
		channel0Single, channel1Single, channel2Single = global_color_hist(imgBGR, spaceType, pathprep_resultDS, imageName)
		save_global_color_hist(channel0Single, channel1Single, channel2Single, dfSingle,spaceType, imageName,pathResults)


#start		
if build_dataset==True:
    # Read Images
    dfDataset = create_df(pathDS)
    
    if prepoces==True:
        for index, row in dfDataset.iterrows():
            queryImage = row["Image"]
            imgBGR = cv2.imread(queryImage,1)
            global_color(imgBGR, spaceType, pathprep_resultDS, imageName)

        store_histogram_total(dfDataset, pathprep_resultDS, spaceType, level=level)

    else:
        # Save image descriptors
        store_histogram_total(dfDataset, pathDS, spaceType, level=level)


if pass_queries == True:
    # Read and store queris images/descriptors
    dfQuery = create_df(pathQueries)
    if prepoces ==True :
        for index, row in dfQuery.iterrows():
            queryImage = row["Image"]
            imgBGR = cv2.imread(queryImage,1)
            global_color(imgBGR, spaceType, pathprep_resultQueries, imageName)

        store_histogram_total(dfQuery,pathprep_resultQueries, spaceType, level=level)
    else:
        store_histogram_total(dfQuery,pathQueries, spaceType, level=level)


    # Create list of lists for all histograms in the dataset  
    whole_hist_list = [histograms_to_list(row_ds, level) for _,row_ds in dfDataset.iterrows() ]
    
    # Compute distance for each query
    # distanceList = list of lists, where each internal list has the 10 lowest distances for each query image
    #distanceList = [getX2results(whole_hist_list,  histograms_to_list(row, level))  for index,row in dfQuery.iterrows() ]
       

# --> MORE HERE <-- #
# --> MORE HERE <-- #
# --> MORE HERE <-- #
# --> MORE HERE <-- #
#
## Texture Descriptors - Haar Wavelets technique + GLCM
#imgTest = get_image(df.iloc[0]['Image'], pathDS)
#grayImg = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
#coeff = haar_wavelet(grayImg, level = 0)
#imgHaar = haar_sticking(coeff, level = 0)
#plot_gray(imgHaar)
#
## Save and Evalaute Results..
#resultTest = pd.DataFrame({
#    'Image' : ['im1', 'im3', 'im67', 'im97', 'im69', 'im46'],
#    'Order' : [2, 1, 0, 1, 0, 2],
#    'Query': [1, 1, 1, 2, 2, 2],
#    })
#
#result_list = submission_list(resultTest)
#
#query_list = result_list
#evaluation = mapk(query_list, result_list, k)
#save_pkl(result_list, pathResults)        