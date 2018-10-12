from ImageFeature import get_ground_truth
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ColorImage import compute_color, compute_histogram_type 
from ColorSegmentation import color_segmentation
#---> KEY PATHS  <----#

testPath = 'datasets/test/'
trainPath = 'datasets/train/'
trainGtPath = 'datasets/train/gt/'
trainMaskPath = 'datasets/train/mask/'
resultsPath = 'm1-results/week1/validation/'

#---> CONFIGURATION  <----#
LOAD_DATA = False
PLOT = True
VALIDATE = False
TEST = 0
MODEL = 2

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

#---> SEGMENTATION  <----#
color_segmentation(dfTest, testPath)

#if validate == True:
#    # Apply filters
#    if model == 1: 
#        color_dict = computeColor(dfValidation, 'HSV', 'mix') 
#        dataset_output_masks = resultsPath    
#        for imageType in color_dict:
#            for image in color_dict[imageType]:
#                name = image[1]
#                cv2.imwrite(dataset_output_masks+name+'.png', image[0])
#    else:
#        if test == 1:
#            colorSegmentation_test(dfTest, testPath)
#        else: 
#            colorSegmentation(dfValidation)
#

#elif validate == "false":
#    # compute histograms for training data from each imageType
#    imageType=["A","B","C","D","E","F"]
#    for i in imageType:
#        hue, sat, val = compute_histogram_type(i, train_dict)
#        #Plot histogram
##        plt.figure(figsize=(10,8))
##        plt.subplot()           
##        plt.subplots_adjust(hspace=.5)
##        plt.title("Hue")
##        plt.hist(hue, bins='auto')
##        plt.subplot(312)                             
##        plt.title("Saturation")
##        plt.hist(sat, bins='auto')
##        plt.subplot(313)
##        plt.title("Luminosity Value")
##        plt.hist(val, bins='auto')
#        
#else: 
#    print ("Wrong entry")