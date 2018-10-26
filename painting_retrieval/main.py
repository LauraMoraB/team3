import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import create_df, get_image, plot_rgb
from method1 import store_histogram_total
from task1 import *

import itertools
# Paths
pathDS = "dataset/"



## Read Images
#df = create_df(pathDS)
### Save image descriptors
#hists= store_histogram_total(df, channel_name=['R','G','B'], level=0)

R = df["level0_R"].iloc[0][:]
G = df["level0_G"].iloc[0][:]
B = df["level0_B"].iloc[0][:]


#print (list(itertools.chain.from_iterable(a)))