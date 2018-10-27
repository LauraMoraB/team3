import cv2
from matplotlib import pyplot as plt
from utils import create_df, submission_list, save_pkl, mapk, get_image, plot_rgb, plot_gray
from method1 import store_histogram_total
from task5 import texture_method1
import pandas as pd
import numpy as np

# Paths
pathDS = "dataset/"
pathQueries = "queries/"
pathResults = "results/"

# Number of results per query
k = 10
# Read Images
df = create_df(pathDS)
# Save image descriptors
store_histogram_total(df, channel_name=['R','G','B'], level=2)

# --> MORE HERE <-- #
# --> MORE HERE <-- #
# --> MORE HERE <-- #
# --> MORE HERE <-- #

# Texture Descriptors - Haar Wavelets technique + GLCM

# EVALUATION
resultList = []
queryList = []
# MAPK RESULT
evaluation = mapk(queryList, resultList, k)
# SAVE RESULTS INTO PKL FILEbins
save_pkl(resultList, pathResults)
