from ImageFeature import getGridOfImage
from ImageSplit import split_by_type
from ImageSplit import divide_dictionary 
from ImageSplit import compute_stats
from create_dataframe import create_df
from create_dataframe import create_df_test
from ColorImage import computeColor
from ColorImage import compute_histogram_type  
from ColorSegmentation import colorSegmentation
from ColorSegmentation import colorSegmentation_test
from collections import defaultdict

import sys 
import cv2
from collections import defaultdict
sys.path.insert(0, 'traffic_signs/')

import traffic_sign_detection

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'

validate = 'false'
test = 0
model = 2

# First, the whole DS is analized a organized in a data structure
# to be used on the next steps
# Then, DS is analized taking into account FR, FF and Area for each signal
try:
    (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot = False)   
except NameError:
    df = create_df(addPath, addPathMask, addPathGt)
    # Dataframe and Dictionary creation
    (image_dict, df) = getGridOfImage(df, addPath, addPathMask, addPathGt)

# Test Dataset
test_df = create_df_test('datasets/test/')
# TRAIN AND VALIDATION
# Second the DS is split into two (70%, 30%) taking into account Size area
(train, validation) = split_by_type(df, addPath,addPathMask ) 

# import into dictionary
(validation_dict, train_dict) = divide_dictionary(image_dict, validation, train)
            
## After the split, the analysis is divided between Training and Validate
if validate == "true":
    # Apply filters
    if model == 1: 
        color_dict = computeColor(validation_dict, "HSV", "mix") 
        dataset_output_masks = "m1-results/week1/validation/"    
        for imageType in color_dict:
            for image in color_dict[imageType]:
                name = image[1]
                cv2.imwrite(dataset_output_masks+name+".png", image[0])
    else:
        if test == 1:
            colorSegmentation_test(test_df, 'datasets/test/')
        else: 
            colorSegmentation(validation_dict)
    
   
elif validate == "false":
    # compute histograms for training data from each imageType
    imageType=["A","B","C","D","E","F"]
    for i in imageType:
        hue, sat, val = compute_histogram_type(i, train_dict)
        #Plot histogram
#        plt.figure(figsize=(10,8))
#        plt.subplot()           
#        plt.subplots_adjust(hspace=.5)
#        plt.title("Hue")
#        plt.hist(hue, bins='auto')
#        plt.subplot(312)                             
#        plt.title("Saturation")
#        plt.hist(sat, bins='auto')
#        plt.subplot(313)
#        plt.title("Luminosity Value")
#        plt.hist(val, bins='auto')
        
else: 
    print ("Wrong entry")