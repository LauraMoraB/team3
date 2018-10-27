import cv2
from utils import  create_df, get_image
from task2 import  hellingerKernel, histIntersection, x2Distance
import matplotlib.pyplot as plt



def getX2results(histogram_list_dataset, histogram_query, K):
    distanceList = []
    
    for histogram_dataset in histogram_list_dataset: 
        distance = x2Distance(histogram_dataset, histogram_query)
        distanceList.append(distance)
    distanceList.sort()
    distanceList =  distanceList[:K]
    return distanceList
        
def getHellingerKernelResult(pathDataset, sourceImageName):
    similarityList = []
    sourceImage = cv2.imread(pathDataset + sourceImageName, 1)
    equalized_hist_source, blue_histr_source, green_histr_source, red_histr_source = getHistograms(sourceImage)

    targetDf = create_df(pathDataset)
    for i in range(len(targetDf)): 
        dfSingle = targetDf.iloc[i]
        targetImage = get_image(dfSingle, pathDataset)
        equalized_hist_target, blue_histr_target, green_histr_target, red_histr_target = getHistograms(targetImage)
        
        similarity =  histIntersection(equalized_hist_target, equalized_hist_source)
        similarityList.append(similarity)
    similarityList.sort()
    return similarityList
    
def getHistInterseccionResult(pathDataset, sourceImageName):
    intersectionList = []
    sourceImage = cv2.imread(pathDataset + sourceImageName, 1)
    equalized_hist_source, blue_histr_source, green_histr_source, red_histr_source = getHistograms(sourceImage)

    targetDf = create_df(pathDataset)
    for i in range(len(targetDf)): 
        dfSingle = targetDf.iloc[i]
        targetImage = get_image(dfSingle, pathDataset)
        equalized_hist_target, blue_histr_target, green_histr_target, red_histr_target = getHistograms(targetImage)
        
        intersection = hellingerKernel(equalized_hist_target, equalized_hist_source)
        intersectionList.append(intersection)
    intersectionList.sort()
    return intersectionList
