import cv2
from matplotlib import pyplot as plt
from utils import create_df, submission_list, save_pkl, mapk, get_image, plot_rgb
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
hists= store_histogram_total(df, channel_name=['R','G','B'], level=1)


# Texture Descriptors - Haar Wavelets technique
imgTest = get_image(df.iloc[5], pathDS)
plt.imshow(imgTest)
plt.show()
grayImg = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)

level = 0
coeff = haar_wavelet(grayImg, level)
imgHaar = haar_sticking(coeff, level)
plt.imshow(imgHaar, cmap='gray')
plt.show()


# Save Results..
dfResult = pd.DataFrame({
    'Image' : ['im1', 'im3', 'im67', 'im97', 'im69', 'im46'],
    'Order' : [2, 1, 0, 1, 0, 2],
    'Query': [1, 1, 1, 2, 2, 2],
    })

result_list = submission_list(dfResult)

query_list = result_list
evaluation = mapk(query_list, result_list, k)


save_pkl(result_list, pathResults)
