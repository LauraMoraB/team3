from ImageFeature import get_ground_truth, save_text_file
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from validation import pixel_validation, validation_window
import numpy as np
from SlidingWindow import window_main
from FastSlidingWindow import fast_sw
import datetime
from argparse import ArgumentParser
from ColorSegmentation import color_segmentation
from ColorSegmentationBinary import color_segmentation_binary
from ColorSegmentationGrey import color_segmentation_grey

from validation import validation

global CONSOLE_ARGUMENTS

def parse_arguments():
    """
	Parse line arguments
	"""
    parser = ArgumentParser()
    general_args = parser.add_argument_group("General arguments")

    general_args.add_argument("-ld", "--load_data", action='store_true', help="Load data")
    general_args.add_argument("-ps", "--plot_slots", action='store_true', help="Ploting slots")

    general_args.add_argument('-t', '--task', choices=('SLW2', 'SLW3', 'SLW_FAST', 'CCL'))

    # create our group of mutually exclusive arguments
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument("--test", action='store_true', help="test excludes train, validate")
    mutually_exclusive.add_argument("--train", action='store_true', help="train excludes test, validate")
    mutually_exclusive.add_argument("--validate", action='store_true', help="validate excludes test, train")

    return parser.parse_args()


CONSOLE_ARGUMENTS = parse_arguments()

#--->  KEY PATHS  <----#
testPath = 'datasets/test/'
fullTrainPath = 'datasets/train/'
validationSplitPath = "datasets/split/validation/"
trainSplitPath = "datasets/split/train/"

#--->  CONFIGURATION  <----#

print(CONSOLE_ARGUMENTS.load_data)
print(CONSOLE_ARGUMENTS.task)
print(CONSOLE_ARGUMENTS.plot_slots)
print(CONSOLE_ARGUMENTS.train)
print(CONSOLE_ARGUMENTS.validate)
print(CONSOLE_ARGUMENTS.test)


LOAD_DATA = False
PLOT_STATS = False
USE_TRAIN = False
USE_VALIDATION = False
USE_TEST = False

# el typeW indicates where images and text files are going to be stored
typeW = "CCL_3"

# The method to be executed
task = "SLW"

if task == "SLW":
    # set to 2 or 3
    method = 3


#--->  COLOR THRESHOLDS  <----#
hsv_rang = (
     np.array([0,150,50]), np.array([20, 255, 255]) #RED
     ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
     ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
)


###--->  DATA PARSING AND SPLIT  <----#
if(LOAD_DATA == True):
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
#
if(USE_TRAIN == True):
    #--->  TRAIN DATA SEGMENTATION  <----#
    listOfBB = color_segmentation(dfTrain, trainSplitPath, hsv_rang)
    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
    pixel_validation(dfTrain, trainSplitPath, 'finalMask')
    
if(USE_VALIDATION == True):
    #--->  VALIDATION DATA SEGMENTATION  <----#
    #listOfBB = color_segmentation(dfValidation, validationSplitPath, hsv_rang)
    
    # amb el path, perteixes de gt o de la nostra segmentacio
    results = "resultMask/finalMask/"
    #results = "mask/"
    
    print ("Start: ", datetime.datetime.now())
    window_canditate =  window_main(dfValidation, validationSplitPath+results, dfStats, typeW, method)
    print ("End: ", datetime.datetime.now())
   # window_canditate =  fast_sw(dfValidation, validationSplitPath, dfStats)
    
#    pixel_validation(dfTrain, trainSplitPath, 'colorMask')
#    pixel_validation(dfTrain, trainSplitPath, 'morphologyMask')
#    pixel_validation(dfTrain, trainSplitPath, 'finalMask')
 
    
    for element in window_canditate:
        name, positions = element
        save_text_file(validationSplitPath+"gtResult/", positions, name, method, typeW)
    
    validation_window(dfValidation, validationSplitPath, typeW )
 
if(USE_TEST == True):
    
    path_complete=testPath+"gtResult/"
    # aixo ha de canviar xk pilli les imatges de test 
    pathToMask = testPath+"resultMask/"
    
    if task == "CCL":    
        #listOfBB = color_segmentation(dfTest, testPath, hsv_rang)
        color_segmentation_binary(dfTest, testPath)
        color_segmentation_grey(dfTest, testPath)

    elif task == "SLW":
        
        init = datetime.datetime.now()
        
        window_canditate =  window_main(dfTest, pathToMask, dfStats, typeW, method)
        
        end = datetime.datetime.now()
        print ("Total Time: ", end-init)
        
        # writing text files as result
        for element in window_canditate:
            name, positions = element
            save_text_file(path_complete, positions, name, method, typeW)
              
    
    elif task == "SLW_FAST":
        
        init = datetime.datetime.now()
        window_canditate =  fast_sw(dfValidation, validationSplitPath, dfStats)
        
        end = datetime.datetime.now()
        print ("Total Time: ", end-init)
        
        # writing text files as result
        for element in window_canditate:
            name, positions = element
            save_text_file(testPath, positions, name, method, typeW)
            
        
    else:
        print ("Entered method is invalid")
    
