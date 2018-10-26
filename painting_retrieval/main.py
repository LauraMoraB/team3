import cv2
from matplotlib import pyplot as plt
from utils import create_df, submission_list, save_pkl, mapk, get_image, plot_rgb, plot_gray
from method1 import store_histogram_total
from task5 import haar_wavelet, haar_sticking
import pandas as pd

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
imgTest = get_image(df.iloc[0]['Image'], pathDS)
grayImg = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
coeff = haar_wavelet(grayImg, level = 0)
imgHaar = haar_sticking(coeff, level = 0)
plot_gray(imgHaar)

# Save and Evalaute Results..
resultTest = pd.DataFrame({
    'Image' : ['im1', 'im3', 'im67', 'im97', 'im69', 'im46'],
    'Order' : [2, 1, 0, 1, 0, 2],
    'Query': [1, 1, 1, 2, 2, 2],
    })

result_list = submission_list(resultTest)

query_list = result_list
evaluation = mapk(query_list, result_list, k)
save_pkl(result_list, pathResults)
