import numpy as np
import pandas as pd
from utils import create_df, submission_list, save_pkl

# Paths
pathDS = "dataset/"
pathQueries = "queries/"
pathResults = "results/"

# Read Images
df = create_df(pathDS)

# Compute Histogram


## ....

# Save Results..
dfResult = pd.DataFrame({
    'Image' : ['im1', 'im3', 'im67', 'im97', 'im69', 'im46'],
    'Order' : [2, 1, 0, 1, 0, 2],
    'Query': [1, 1, 1, 2, 2, 2],
    })

result_list = submission_list(dfResult)
save_pkl(result_list, pathResults)