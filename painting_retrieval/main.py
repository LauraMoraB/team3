from utils import create_df 
from task1 import color_characterization, compute_histograms

# Paths
pathDS = "dataset/museum_set_random/"
pathDSquery = "query_devel/query_devel_random/"

# Read Images
df = create_df(pathDS)

# Compute Histogram
dfq = create_df(pathDSquery)
color_characterization(df, pathDS)
compute_histograms(df, pathDS,'HSV')


## ....

# Save Results..
