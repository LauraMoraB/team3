from utils import get_window_gt
import numpy as np
#import sys
#sys.path.insert(0, "../TrafficSigns/traffic_signs/evaluation/")
#from bbox_iou import bbox_iou

def bbox_iou(bboxA, bboxB):
   
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])
    

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both bboxes
    bboxAArea = (bboxA[2] - bboxA[0] + 1) * (bboxA[3] - bboxA[1] + 1)
    bboxBArea = (bboxB[2] - bboxB[0] + 1) * (bboxB[3] - bboxB[1] + 1)
    
    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    return iou

### WINDOW EVALUATION
def performance_accumulation_window(detections, annotations):
    detections_used  = np.zeros(len(detections))
    annotations_used = np.zeros(len(annotations))
    annotations = [annotations]
    detections=[detections]
    
    TP = 0
    if len(detections[0]) != 0:
        for ii in range (len(annotations)):
            for jj in range (len(detections)):
                
                if (detections_used[ii] == 0) & (bbox_iou(annotations[ii], detections[jj]) > 0.5):
                    TP = TP+1
                    detections_used[jj]  = 1
                    annotations_used[ii] = 1

    FN = np.sum(annotations_used==0)
    FP = np.sum(detections_used==0)

    return [TP,FN,FP]

def performance_evaluation_window(TP, FP, FN):

    precision   = float(TP) / float(TP+FP);
    sensitivity = float(TP) / float(TP+FN)
    accuracy    = float(TP) / float(TP+FN+FP);

    return [precision, sensitivity, accuracy]

def validation_window(pathGT, pathResults):
    """
    Compute performance of the model
    """
    bboxesGT = get_window_gt(pathGT)
    bboxesResult = get_window_gt(pathResults)

    TruePos = 0
    FalsePos = 0
    FalseNeg = 0

    for i in range(len(bboxesGT)):
        
        [pixelTP, pixelFN, pixelFP] = performance_accumulation_window(bboxesResult[i], bboxesGT[i])
    
        TruePos += pixelTP
        FalsePos += pixelFP
        FalseNeg += pixelFN
        
            
    [precision, sensitivity, accuracy] = performance_evaluation_window(TruePos, FalsePos, FalseNeg)
    
    print("\n------ FINAL RESULT Window Validation -------")
    print('Window Precision: ', precision)
    print('Accuracy: ', accuracy)
    print('Window Sensitivity: ', sensitivity)
    
    print (TruePos, FalsePos, FalseNeg)
    #F1 = compute_f1(precision, sensitivity)

    #print('F1: ', F1)

   