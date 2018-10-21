from ImageFeature import get_ground_truth, save_text_file
from ImageSplit import split_by_type, compute_stats, plot_stats
from createDataframe import create_df_train, create_df_test
from ImageSegmentation import color_segmentation
from validation import pixel_validation, validation_window
import numpy as np
from SlidingWindow import window_main
from FastSlidingWindow import fast_sw
import datetime

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
USE_TEST = False

#--->  COLOR THRESHOLDS  <----#
hsv_rang = (
     np.array([0,150,50]), np.array([20, 255, 255]) #RED
     ,np.array([160,150,50]), np.array([180, 255, 255]) #DARK RED
     ,np.array([100,150,50]), np.array([140, 255, 255]) #BLUE
)

# el typeW indicates where images and text files are going to be stored
typeW = "SLW_GT_SQUARE_TEST"

task = "SLW"

if task == "SLW":
    # set to 2 or 3
    method = 1

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
    
    if task == "CCL":    
        listOfBB = color_segmentation(dfTest, testPath, hsv_rang)
    
    elif task == "SLW":
        
        init = datetime.datetime.now()
        
        window_canditate =  window_main(dfTest, path_complete, dfStats, typeW, method)
        
        end = datetime.datetime.now()
        print ("Total Time: ", end-init)
        
        # writing text files as result
        for element in window_canditate:
            name, positions = element
            save_text_file(path_complete, positions, name, method, typeW)
        
        validation_window(dfTest, path_complete, typeW )
      
    
    elif task == "SLW_FAST":
        
        init = datetime.datetime.now()
        window_canditate =  fast_sw(dfValidation, validationSplitPath, dfStats)
        
        end = datetime.datetime.now()
        print ("Total Time: ", end-init)
        
        # writing text files as result
        for element in window_canditate:
            name, positions = element
            save_text_file(testPath, positions, name, method, typeW)
            
        validation_window(dfTest, path_complete, typeW )
        
    else:
        print ("Entered method is invalid")
    
    

    