from ImageFeature import get_ground_truth
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ImageSegmentation import color_segmentation
from Validation import pixel_validation

#---> KEY PATHS  <----#
testPath = 'datasets/test/'
fullTrainPath = 'datasets/train/'
validationSplitPath = "datasets/split/validation/"
trainSplitPath = "datasets/split/train/"

#---> CONFIGURATION  <----#
LOAD_DATA = False
PLOT_STATS = False
USE_TRAIN = True
USE_VALIDATION = False

#---> DATA PARSING AND SPLIT  <----#
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
    #---> TRAIN DATA SEGMENTATION  <----#
    listOfBB = color_segmentation(dfTrain, trainSplitPath)
    pixel_validation(dfTrain, trainSplitPath)
    
if(USE_VALIDATION == True):
    #---> VALIDATION DATA SEGMENTATION  <----#
    listOfBB = color_segmentation(dfValidation, validationSplitPath)
    pixel_validation(dfValidation, validationSplitPath)