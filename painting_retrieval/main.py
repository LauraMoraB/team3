import cv2
from matplotlib import pyplot as plt
from utils import create_df, submission_list, save_pkl, mapk, get_image, plot_rgb, plot_gray
from method1 import store_histogram_total, histograms_to_list
from task5 import haar_wavelet, haar_sticking
from task3 import getX2results, getHellingerKernelResult, getHistInterseccionResult
import pandas as pd

# Paths
pathDS = "dataset/"
pathQueries = "queries/"
pathResults = "results/"

# Number of results per query
k = 10
build_dataset=False
pass_queries=True
level=0
channel_name="BGR"


if build_dataset==True:
    # Read Images
    dfDataset = create_df(pathDS)
    
    # Save image descriptors
    store_histogram_total(dfDataset, pathDS, channel_name, level=level)
    
if pass_queries == True:
    X2resultList = []
    HIresultList = []
    HKresultList = []

    queryList = []
    # Read and store queris images/descriptors
    dfQuery = create_df(pathQueries)
    store_histogram_total(dfQuery,pathQueries, channel_name, level=level)
 
    whole_hist_list = [histograms_to_list(row_ds, level) for _,row_ds in dfDataset.iterrows() ]
    for index,row in dfQuery.iterrows():
        histogram_query = histograms_to_list(row, level)
        queryList.append(row['Image'])
        
        X2resultList.append(getX2results(whole_hist_list, histogram_query,  k, dfDataset))
        HIresultList.append(getHistInterseccionResult(whole_hist_list, histogram_query,  k, dfDataset))
        HKresultList.append(getHellingerKernelResult(whole_hist_list, histogram_query,  k, dfDataset))

#    distanceList = [getX2results(whole_hist_list,  histograms_to_list(row, level), k)  for index,row in dfQuery.iterrows() ]
       

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
