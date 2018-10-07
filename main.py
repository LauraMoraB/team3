from ImageFeature import getGridOfImage
from ImageSplit import split_by_type 
from ImageSplit import compute_stats
from create_dataframe import create_df
from create_dataframe import create_df_test
from ColorImage import computeColor
from ColorImage import compute_histogram_type  
from ColorSegmentation import colorSegmentation
from ColorSegmentation import colorSegmentation_test
import sys 
import cv2
from collections import defaultdict
sys.path.insert(0, 'traffic_signs/')

import traffic_sign_detection

addPath = 'datasets/train/'
addPathGt = 'datasets/train/gt/'
addPathMask = 'datasets/train/mask/'
validate = 'true'
model = 2

def divide_dictionary(dictionary, dataFrame1, dataFrame2):
    dict1 = defaultdict(list)
    dict2 = defaultdict(list)
    for typeSignal in dictionary:
        type1 = dataFrame1[dataFrame1.Type == typeSignal]
        typeName1 = type1.Image.values.tolist()
        typeArea1 = type1.Area.values.tolist()
        type2 = dataFrame2[dataFrame2.Type == typeSignal]
        typeName2 = type2.Image.values.tolist()
        typeArea2 = type2.Area.values.tolist()
        for signal in image_dict[typeSignal]:
            if signal.name+'.jpg' in typeName1 and signal.area in typeArea1:
                dict1[typeSignal].append(signal)
            elif signal.name+'.jpg' in typeName2 and signal.area in typeArea2:
                dict2[typeSignal].append(signal)    
    return (dict1, dict2)


def read_test():
    test = create_df_test('datasets/test/')
    return test   
 
# First, the whole DS is analized a organized in a data structure
# to be used on the next steps
# Then, DS is analized taking into account FR, FF and Area for each signal
test_df = read_test()

try:
    (fillRatioStats, formFactorStats, areaStats) = compute_stats(image_dict, plot = False)   
except NameError:
    df = create_df(addPath, addPathMask, addPathGt)
    # Dataframe and Dictionary creation
    (image_dict, df) = getGridOfImage(df, addPath, addPathMask, addPathGt)

# Second the DS is split into two (70%, 30%) taking into account Size area
(train, validation) = split_by_type(df) 

# import into dictionary
(validation_dict, train_dict) = divide_dictionary(image_dict, validation, train)
            
# After the split, the analysis is divided between Training and Validate
if validate == "true":
    # Apply filters
    if model == 1: 
        ############
        ## MODEL 1 #
        ############
        color_dict = computeColor(validation_dict, "HSV", "mix")
        ## Evaluate Model
        # Save images
        # Directori amb els resultats obtinguts
        dataset_output_masks = "m1-results/week1/validation/senseResta"    
        for imageType in color_dict:
            for image in color_dict[imageType]:
                name = image[1]
                cv2.imwrite(dataset_output_masks+name+".jpg", image[0])
    else:
        ############
        ## Model 2 #
        ############
        colorSegmentation_test(test_df, 'datasets/test/')
        #colorSegmentation(validation_dict)
    
   
elif validate == "false":
    # compute histograms for training data from each imageType
    imageType="B"
    hue, sat, val = compute_histogram_type(imageType)
else: 
    print ("Wrong entry")