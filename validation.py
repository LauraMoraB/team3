import numpy as np
from ImageFeature import get_full_mask, get_full_mask_result
from createDataframe import load_annotations
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "traffic_signs/evaluation/")
from bbox_iou import bbox_iou

def performance_accumulation_window(detections, annotations):

    detections_used  = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))
    
    TP = 0
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

def validation_window(bboxes, pathToGT):
    # in this case, the list of possible bboxes is returned as a list
    TruePos = 0
    FalsePos = 0
    FalseNeg = 0

    for i in range(len(bboxes)):
        name, lista = bboxes[i]
        for x in lista:
            # First window of the possible detections
            windows = x.copy()
            # In the first place of the list, the image name has to be stored so we can get the name
            gtFile = "gt."+name+".txt"
    
            annotations = load_annotations(pathToGT+'gt/'+gtFile)
    
            # remove image name to evaluate position
            [pixelTP, pixelFN, pixelFP] = performance_accumulation_window(windows, annotations)


            TruePos += pixelTP
            FalsePos += pixelFP
            FalseNeg += pixelFN
            
    [precision, sensitivity, accuracy] = performance_evaluation_window(TruePos, FalseNeg, FalsePos)
    
    print("\n------ FINAL RESULT Window Validation -------")
    print('Window Precision: ', precision)
    print('Accuracy: ', accuracy)
    print('Window Sensitivity: ', sensitivity)

    F1 = compute_f1(precision, sensitivity)

    print('F1: ', F1)

    return [TruePos, FalsePos, FalseNeg]
