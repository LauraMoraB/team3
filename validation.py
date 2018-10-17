import numpy as np
from ImageFeature import get_full_mask, get_full_mask_result
from createDataframe import load_annotations
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "traffic_signs/evaluation/")
from bbox_iou import bbox_iou


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

    pixel_F1 = compute_f1(pixel_precision, pixel_sensitivity)


    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, pixel_F1]


### WINDOW EVALUATION
def performance_accumulation_window(detections, annotations):
    """
    performance_accumulation_window()

    Function to compute different performance indicators (True Positive,
    False Positive, False Negative) at the object level.

    Objects are defined by means of rectangular windows circumscribing them.
    Window format is [ struct(x,y,w,h)  struct(x,y,w,h)  ... ] in both
    detections and annotations.

    An object is considered to be detected correctly if detection and annotation
    windows overlap by more of 50%

       function [TP,FN,FP] = PerformanceAccumulationWindow(detections, annotations)

       Parameter name      Value
       --------------      -----
       'detections'        List of windows marking the candidate detections
       'annotations'       List of windows with the ground truth positions of the objects

    The function returns the number of True Positive (TP), False Positive (FP),
    False Negative (FN) objects
    """

    detections_used  = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))

    TP = 0
    print (detections)
    for ii in range (len(annotations)):
        #for jj in range (len(detections)):

        if (detections_used[ii] == 0) & (bbox_iou(annotations[ii], detections) > 0.5):
            TP = TP+1
            detections_used[ii]  = 1
            annotations_used[ii] = 1

    FN = np.sum(annotations_used==0)
    FP = np.sum(detections_used==0)

    return [TP,FN,FP]

def performance_evaluation_window(TP, FN, FP):
    """
    performance_evaluation_window()

    Function to compute different performance indicators (Precision, accuracy,
    sensitivity/recall) at the object level

    [precision, sensitivity, accuracy] = PerformanceEvaluationPixel(TP, FN, FP)

       Parameter name      Value
       --------------      -----
       'TP'                Number of True  Positive objects
       'FN'                Number of False Negative objects
       'FP'                Number of False Positive objects

    The function returns the precision, accuracy and sensitivity
    """

    precision   = float(TP) / float(TP+FP); # Q: What if i do not have TN?
    sensitivity = float(TP) / float(TP+FN)
    accuracy    = float(TP) / float(TP+FN+FP);

    return [precision, sensitivity, accuracy]

def mask_visual_check(df, path):
    for i in range(len(df)):
        dfSingle = df.iloc[i]
        maskResult = get_full_mask_result(dfSingle, path)
        plt.figure()
        plt.imshow(maskResult)
        maskValidator = get_full_mask(dfSingle, path)
        plt.figure()
        plt.imshow(maskValidator)

def compute_f1(precision, recall):

    if(precision == 0 or recall == 0):
        F1 = 0
    else:
       F1 =  2*(float(precision)*float(recall))/(float(recall) + float(precision))

    return F1

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


def validation_window(bboxes, pathToGT):
    # in this case, the list of possible bboxes is returned as a list
    TruePos = 0
    FalsePos = 0
    FalseNeg = 0

    for i in range(len(bboxes)):
        # First window of the possible detections
        windows = bboxes[i]
        # In the first place of the list, the image name has to be stored so we can get the name
        gtFile = "gt."+windows[0]+".txt"

        annotations = load_annotations(pathToGT+'gt/'+gtFile)

        # remove image name to evaluate position
        windows.pop(0)
        [pixelTP, pixelFP, pixelFN] = performance_accumulation_window(windows, annotations)

        TruePos += pixelTP
        FalsePos += pixelFP
        FalseNeg += pixelFN

    [precision, sensitivity, accuracy] = performance_evaluation_window(TruePos, FalsePos, FalseNeg)

    print('Window Precision: ', precision)
    print('Accuracy: ', accuracy)
    print('Window Sensitivity: ', sensitivity)

    F1 = compute_f1(precision, sensitivity)

    print('F1: ', F1)

    return [TruePos, FalsePos, FalseNeg]