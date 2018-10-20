from ImageFeature import get_ground_truth, save_text_file
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ImageSegmentation import color_segmentation
from validation import pixel_validation, validation_window
import numpy as np
from SlidingWindow import window_main
from FastSlidingWindow import fast_sw

#--->  KEY PATHS  <----#
testPath = 'datasets/test/'
fullTrainPath = 'datasets/train/'
validationSplitPath = "datasets/split/validation/"
trainSplitPath = "datasets/split/train/"

#--->  CONFIGURATION  <----#
LOAD_DATA = True
PLOT_STATS = False
USE_TRAIN = False
USE_VALIDATION = True

#--->  COLOR THRESHOLDS  <----#
hsv_rang = (
     np.array([0,150,50]), np.array([20, 255, 255]) #RED
     ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
     ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
)
# ---> METHOD TO SAVE GT <--- #
typeW = "SLW"
method = 1

#--->  DATA PARSING AND SPLIT  <----#
if(LOAD_DATA == False):
# df is created by Parsing training image folders
    df = create_df_train(fullTrainPath)
    # df is updated computing provided groundtruth informa2tion
    df = get_ground_truth(df, fullTrainPath)
    # df is created with test images
    dfTest = create_df_test(testPath)
    # stats are worked out over the df
    dfStats = compute_stats(df)
    # ds is splited into two sets (70%, 30%) taking into account signal area size
    (dfTrain, dfValidation) = split_by_type(df, fullTrainPath)

if(PLOT_STATS == True):
    plot_stats(df)

if(USE_TRAIN == True):
    #--->  TRAIN DATA SEGMENTATION  <----#
    listOfBB = color_segmentation(dfTrain, trainSplitPath, hsv_rang)
    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
    pixel_validation(dfTrain, trainSplitPath, 'finalMask')
    validation_window(listOfBB, trainSplitPath)
    
if(USE_VALIDATION == True):
    #--->  VALIDATION DATA SEGMENTATION  <----#
    #listOfBB = color_segmentation(dfValidation, validationSplitPath, hsv_rang)
    
    window_canditate =  window_main(dfValidation, validationSplitPath+"resultMask/morphologyMask/", dfStats)
    window_canditate =  fast_sw(dfValidation, validationSplitPath, dfStats)
    
#    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
#    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
#    pixel_validation(dfTrain, trainSplitPath, 'finalMask')
#    
    validation_window(window_canditate, validationSplitPath )
    
    
    for element in window_canditate:
        name, positions = element
        save_text_file(validationSplitPath+"gtResult/", positions, name, method, typeW)
    

# SLIDING WINDOWS SPEED COMPARISON
    

    
##### Window #####
#window_canditate =  window_main(dfTrain, trainSplitPath)
#validation_window(window_canditate, trainSplitPath )
#for element in window_canditate:
#    name, positions = element
#    save_text_file(trainSplitPath+"gtResult/", positions, name)
