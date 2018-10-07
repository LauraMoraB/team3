# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:04:39 2018

@author: Aitor Sanchez
"""

from ImageFeature import getGridOfImage
from ImageFeature import compute_histogram_type  
from ImageSplit import split_by_type 
from ImageSplit import compute_stats
from create_dataframe import create_df
from ColorImage import computeColor

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'
validate = 'true'

# First, the whole DS is analized a organized in a data structure
# to be used on the next steps
# Then, DS is analized taking into account FR, FF and Area for each signal

try:
    (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot = False)   
except NameError:
    df = create_df(addPath, addPathMask, addPathGt)
    # Dataframe and Dictionary creation
    (image_dict, df) = getGridOfImage(df, addPath, addPathMask, addPathGt)

# Second the DS is split into two (70%, 30%) taking into account Size area
(train, validation) = split_by_type(df)

# After the split, the analysis is divided between Training and Validate
if validate == "true":
    # Apply filters
    #for image in validation.Image:
     #   computeColor()
    # Post-Processing Result
    
    # Decide if canditate is valid
    
    # Evaluate Model
    print ("hi")
elif validate == "false":
    # compute histograms for training data from each imageType
    imageType="B"
   # hue, sat, val = compute_histogram_type(imageType)
    
   
else: 
    print ("Wrong entry")