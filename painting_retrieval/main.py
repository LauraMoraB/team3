import cv2
import numpy as np

from utils import create_df, get_full_image, plot_rgb
from task1 import color_characterization, compute_histograms, divide_image
from method1 import store_histogram_total
# Paths
pathDS = "dataset/museum_set_random/"
pathDSquery = "query_devel/query_devel_random/"

# Read Images
df = create_df(pathDS)

#ch1="HistR"
#ch3="HistB"
#ch2="HistG"
#df = store_histogram_total(df, ch1, ch2, ch3,pathDS)
#hists= store_histogram_total(df, channel_name=['R','G','B'], level=1)


# Compute HistogramGaby
dfq = create_df(pathDSquery)
#color_characterization(df, pathDS, 'HSV')
color_characterization(df, pathDS, 'HSV', "HLS")

#compute_histograms(df, pathDS,'HSV')


## ....
## divide Image in 4 regions LAURA
#hists= store_histogram_total(df, channel_name=['R','G','B'], level=1)
#
#dfSingle = df.iloc[2]
#im = get_full_image(dfSingle, pathDS)
#portions = divide_image(im, 2)
## compute hist for those regions


# Save Results..
# save results
#tuple (image-name, listOfList)
#listOfPositions = [ [startX, startY, endX, endY], [startX2, startY2, endX2, endY2],... ]
#listOfHistogram = [ [[r],[g],[b]] , [[r2],[g2],[b2]] ]  
# 