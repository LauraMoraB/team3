import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import create_df, get_image, plot_rgb
from method1 import store_histogram_total

# Paths
pathDS = "dataset/"



# Read Images
df = create_df(pathDS)
# ch1="HistR"
# ch3="HistB"
# ch2="HistG"
#df = store_histogram_total(df, ch1, ch2, ch3)

# divide Image in 4 regions
im = get_image(df["Image"].iloc[2], pathDS)
portions = divide_image(im, 2)
# compute hist for those regions

# save results
# tuple (image-name, listOfList)
# listOfPositions = [ [startX, startY, endX, endY], [startX2, startY2, endX2, endY2],... ]
# listOfHistogram = [ [[r],[g],[b]] , [[r2],[g2],[b2]] ]  
#   