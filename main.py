from ImageFeature import getGridOfImage
from ImageSplit import split_by_type 
from ImageSplit import compute_stats
from create_dataframe import create_df
from ColorImage import computeColor
from ColorImage import compute_histogram_type  
import sys 
import cv2
from collections import defaultdict
sys.path.insert(0, 'traffic_signs/')

import traffic_sign_detection

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

# import into dictionary
validation_dict = defaultdict(list)
train_dict = defaultdict(list)

for typeSignal in image_dict:
    typeTrain = train[train.Type == typeSignal]
    typeTrainList = typeTrain.Image.values.tolist()
    typeValid = validation[validation.Type == typeSignal]
    typeValidList = typeValid.Image.values.tolist()
    
    for signal in image_dict[typeSignal]:
    
        if signal.name+'.jpg' in typeTrainList:
            train_dict[typeSignal].append(signal)
        
        elif signal.name+'.jpg' in typeValidList:
            validation_dict[typeSignal].append(signal)
            
# After the split, the analysis is divided between Training and Validate
if validate == "true":
    # Apply filters
    # from DF to Dictionary
    color_dict = computeColor(validation_dict, "HSV", "mix")
    
    # Post-Processing Result ?
    
    # Decide if canditate is valid
    
    ## Evaluate Model
    # Save images
    # Directori amb els resultats obtinguts
    dataset_output_masks = "m1-results/week1/validation/senseResta"
    
    for imageType in color_dict:
        for image in color_dict[imageType]:
            name = image[1]
            cv2.imwrite(dataset_output_masks+name+".jpg", image[0])

    # Directori amb les imatges reals + annotacions
    dataset_image_directory = "datasets/train/validation/"
    
   
elif validate == "false":
    # compute histograms for training data from each imageType
    imageType="B"
    hue, sat, val = compute_histogram_type(imageType)
else: 
    print ("Wrong entry")