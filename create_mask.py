import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
from create_dataframe import create_df

# Create Dataframe
path_images = "./datasets/train/"
path_masks = "./datasets/train/mask/"
path_gt = "./datasets/train/gt/"
df = create_df(path_images,path_masks,path_gt)

img = cv2.imread('./datasets/train/'+df.get_value(0, 'Image'),1)
mask = cv2.imread('./datasets/train/mask/'+df.get_value(0, 'Mask'),0)
masked_img = cv2.bitwise_and(img,img,mask = mask)

# show image in color
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# show mask
plt.imshow(mask, cmap = 'binary', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
# show masked image
plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
plt.show()