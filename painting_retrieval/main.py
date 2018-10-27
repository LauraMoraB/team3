import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import create_df, get_full_image, submission_list, save_pkl, mapk, get_image, plot_rgb, plot_gray
from method1 import store_histogram_total, histograms_to_list
from task5 import haar_wavelet, haar_sticking
from global_color_histograms import global_color_hist,save_global_color_hist, global_color
import pandas as pd

# Paths
pathDS = "dataset/"
pathQueries = "queries/query_devel_random/"
pathResults = "results/"
pathprep_resultDS = "results_preprocesadoDS/"
pathprep_resultQueries = "results_preprocesadoQueries/"


# Number of results per query
k = 10
build_dataset=True
pass_queries=False
level=2
#type of space 
spaceType= "HSV" #"HSV", "HSL","LAB", "YCrCb","XYZ","LUV"

# Read Images
dfDataset = create_df(pathDS)
dfQuery = create_df(pathQueries)

#choose prepoces
prepoces =True


#choose global_color_histograms: image will be procesed and change space color and save global_color_hist in resuts_GVHistogram (create file )

global_color_histograms =False

if global_color_histograms==True:
	for i in range(len(dfDataset)):       
		dfSingle = dfDataset.iloc[i]
		imgBGR = get_full_image(dfSingle, pathDS)    
		imageName = dfSingle['Image']  
		channel0Single, channel1Single, channel2Single = global_color_hist(imgBGR, spaceType, pathprep_resultDS, imageName)
		save_global_color_hist(channel0Single, channel1Single, channel2Single, dfSingle,spaceType, imageName,pathResults)



#start		
if build_dataset==True:
   if prepoces==True:
      for i in range(len(dfDataset)):
          dfSingle = dfDataset.iloc[i]
          imgBGR = get_full_image(dfSingle, pathDS)
          imageName = dfSingle['Image']
          global_color(imgBGR, spaceType, pathprep_resultDS, imageName)
#      store_histogram_total(dfDataset, pathprep_resultDS, spaceType, level=level)
#	else:
#		## Save image descriptors
#      store_histogram_total(dfDataset, pathDS, spaceType, level=level)


   
if pass_queries == True :
    # Despres s'ha de cridar la funcio que calcula distancies
    imageName="ima_000000.jpg"
    queryImage=pathQueries+imageName
    if prepoces ==True :
        imgBGR = cv2.imread(queryImage,1)
        global_color(imgBGR, spaceType, pathprep_resultQueries, imageName)

        store_histogram_total(dfQuery,pathprep_resultQueries, spaceType, level=level)
    else:
        store_histogram_total(dfQuery,pathQueries, spaceType, level=level)
    
#    histogram_list_query = histograms_to_list(dfQuery, level, 0)
#    for i in range(1):#(len(df)):
#        histogram_list_dataset = histograms_to_list(dfDataset, level, i)
		# S'ha de retocar xk accepti aixo
		#distanceList = getX2results(histogram_list_dataset,  histogram_list_query)


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