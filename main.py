from ImageFeature import get_ground_truth
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ColorSegmentation import color_segmentation
from ColorSegmentationBinary import color_segmentation_binary
from ColorSegmentationGrey import color_segmentation_grey

from validation import validation

#---> KEY PATHS  <----#

testPath = 'datasets/test/'
trainPath = 'datasets/train/'
trainGtPath = 'datasets/train/gt/'
trainMaskPath = 'datasets/train/mask/'
resultsPath = 'm1-results/week1/validation/'
pathToResults = "datasets/split/train/result/"
pathToMasks = "datasets/split/train/mask/"
validationSplitPath = "datasets/split/validation/"
trainSplitPath = "datasets/split/train/"

#---> CONFIGURATION  <----#
LOAD_DATA = False
PLOT = False
FULL_TRAIN = False
FULL_VALIDATION = False
TEST = True

#---> DATA PARSING AND SPLIT  <----#
if(LOAD_DATA == True):
# df is created by Parsing training image folders
    df = create_df_train(trainPath, trainMaskPath, trainGtPath)
    # df is updated computing provided groundtruth information
    df = get_ground_truth(df, trainPath)
    # df is created with test images
    dfTest = create_df_test(testPath)
    # stats are worked out over the df
    dfStats = compute_stats(df)    
    # ds is splited into two sets (70%, 30%) taking into account signal area size
    (dfTrain, dfValidation) = split_by_type(df, trainPath, trainMaskPath) 
    
if(PLOT == True):
    plot_stats(df)
    
if(FULL_VALIDATION == True):
    #---> VALIDATION DATA SEGMENTATION  <----#
    listOfBB = color_segmentation(dfValidation, validationSplitPath)
    validation(dfValidation, validationSplitPath)

if(FULL_TRAIN == True):
    #---> TRAIN DATA SEGMENTATION  <----#
    color_segmentation(dfTrain, trainSplitPath)
    validation(dfTrain, trainSplitPath)
    
if(TEST == True):
#    color_segmentation_binary(dfTest, testPath)
    color_segmentation_grey(dfTest, testPath)
