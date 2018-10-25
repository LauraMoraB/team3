import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import create_df, get_image, plot_rgb
from method1 import store_histogram_total
from task1 import *

# Paths
pathDS = "dataset/"



# Read Images
df = create_df(pathDS)
# Save image descriptors
hists= store_histogram_total(df, channel_name=['R','G','B'], level=1)