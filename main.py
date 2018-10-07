# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:04:39 2018

@author: Aitor Sanchez
"""

from ImageFeature import getGridOfImage 
from ImageSplit import split_by_type 
from ImageSplit import compute_stats
from create_dataframe import create_df


addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

# First, the whole DS is analized a organized in a data structe
# to be used on the next steps

# Dataframe and Dictionary creation
df = create_df(addPath, addPathMask, addPathGt)
(image_dict, df) = getGridOfImage(df, addPath, addPathMask, addPathGt)

# DS is analized taking into account FR, FF and Area for each signal
plot = False
(fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot)

# Second the DS is split into two (70%, 30%) taking into account Size area
(train, validation) = split_by_type(df)

