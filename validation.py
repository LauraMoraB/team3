import numpy as np
from ImageFeature import get_full_mask, get_full_mask_result
from matplotlib import pyplot as plt


def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """ 
    performance_accumulation_pixel()

    Function to compute different performance indicators 
    (True Positive, False Positive, False Negative, True Negative) 
    at the pixel level
       
    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)
       
    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the detected areas
    'pixel_annotation'   Binary image containing ground truth
       
    The function returns the number of True Positive (pixelTP), False Positive (pixelFP), 
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """
    
    pixel_candidates = np.uint64(pixel_candidates>0)
    pixel_annotation = np.uint64(pixel_annotation>0)
    
    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation==0))
    pixelFN = np.sum((pixel_candidates==0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates==0) & (pixel_annotation==0))


    return (pixelTP, pixelFP, pixelFN, pixelTN)
 
def performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN):
    """
    performance_evaluation_pixel()

    Function to compute different performance indicators (Precision, accuracy, 
    specificity, sensitivity) at the pixel level
    
    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = PerformanceEvaluationPixel(pixelTP, pixelFP, pixelFN, pixelTN)
    
       Parameter name      Value
       --------------      -----
       'pixelTP'           Number of True  Positive pixels
       'pixelFP'           Number of False Positive pixels
       'pixelFN'           Number of False Negative pixels
       'pixelTN'           Number of True  Negative pixels
    
    The function returns the precision, accuracy, specificity and sensitivity
    """
    
    pixel_precision   = float(pixelTP) / float(pixelTP+pixelFP)
    pixel_accuracy    = float(pixelTP+pixelTN) / float(pixelTP+pixelFP+pixelFN+pixelTN)
    pixel_specificity = float(pixelTN) / float(pixelTN+pixelFP)
    pixel_sensitivity = float(pixelTP) / float(pixelTP+pixelFN)
    if(pixel_sensitivity == 0 or pixel_precision == 0):
        pixel_F1 = 0
    else:
        pixel_F1 = 2*(float(pixel_sensitivity)*float(pixel_precision))/(float(pixel_sensitivity) + float(pixel_precision))

    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1]

      
def pixel_validation(df, pathGT, maskType):    

    TruePos = 0
    FalsePos = 0
    FalseNeg = 0
    TrueNeg = 0
    for i in range(len(df)):       
        dfSingle = df.iloc[i]

        maskResult = get_full_mask_result(dfSingle, pathGT, maskType)
        maskValidator = get_full_mask(dfSingle, pathGT)
        
        [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(maskResult, maskValidator)
        TruePos += pixelTP
        FalsePos += pixelFP
        FalseNeg += pixelFN
        TrueNeg += pixelTN
        
    [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1] = performance_evaluation_pixel(TruePos, FalsePos, FalseNeg, TrueNeg)
    print('RESULTS -> using ',maskType)
    print('Precission: ', pixel_precision)
    print('Accuracy: ', pixel_accuracy)
    print('Specificity: ', pixel_specificity)
    print('Recall: ', pixel_sensitivity)
    print('F1: ', pixel_F1)

    return [TruePos, FalsePos, FalseNeg, TrueNeg]

def mask_visual_check(df, path):
    for i in range(len(df)):       
        dfSingle = df.iloc[i]
        maskResult = get_full_mask_result(dfSingle, path)
        plt.figure()
        plt.imshow(maskResult)
        maskValidator = get_full_mask(dfSingle, path)
        plt.figure()
        plt.imshow(maskValidator)        
        
    
    
        

            