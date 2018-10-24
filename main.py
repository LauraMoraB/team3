from ImageFeature import get_ground_truth
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ImageSegmentation import color_segmentation
from validation import pixel_validation, validation_window
#from TemplateMatching_use_mask import creattemplate_GRIS#mediatemplate#, creattemplate_GRIS
from Matching import Matching_GRIS
import numpy as np

#--->  KEY PATHS  <----#
testPath = 'datasets/test/'
fullTrainPath = 'datasets/train/'
validationSplitPath = "datasets/split/validation/"
trainSplitPath = "datasets/split/train/"

#--->  CONFIGURATION  <----#


LOAD_DATA = False
PLOT_STATS = False
USE_TRAIN = False
USE_VALIDATION = True

#--->  COLOR THRESHOLDS  <----#
#ImageSegmentation
#--->  DATA PARSING AND SPLIT  <----#
if(LOAD_DATA == True):
# df is created by Parsing training image folders
    df = create_df_train(fullTrainPath)
    # df is updated computing provided groundtruth information
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
    
#    get_color_histogram_GT(dfTrain, trainSplitPath)
#    mediatemplate(df, fullTrainPath)
#    print("media ok")
#    creattemplate_GRIS(df, fullTrainPath)
#    print("MATCHING ok")
    #Matching_GRIS(df, fullTrainPath)

    print("MATCHING ok")
    #creattemplate(dfTrain, trainSplitPath)
    listOfBB = color_segmentation(dfTrain, trainSplitPath)
    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
    pixel_validation(dfTrain, trainSplitPath, 'finalMask')

if(USE_VALIDATION == True):
    #--->  VALIDATION DATA SEGMENTATION  <----#
    Matching_GRIS(dfValidation, validationSplitPath)
    print("MATCHING ok")
#    listOfBB = color_segmentation(dfValidation, validationSplitPath)
#    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
#    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
#    pixel_validation(dfTrain, trainSplitPath, 'finalMask')
##### Window #####
#window_main(dfTrain, trainSplitPath)
#window_candidate = [['00.001147', 70, 183, 139, 251, 'E'],['00.001150', 70, 183, 139, 251, 'F']]
#validation_window(window_candidate, validationSplitPath )