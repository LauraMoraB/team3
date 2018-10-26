import cv2
from utils import  create_df, get_image
from task2 import  hellingerKernel, histIntersection, x2Distance
import matplotlib.pyplot as plt



def getHistograms(image):
    greyImage = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    equalized = cv2.equalizeHist(greyImage )
    equalized_hist = cv2.calcHist([equalized],[0],None,[256],[0,256])
    
    blue_histr = cv2.calcHist([image],[0],None,[256],[0,256])
    green_histr = cv2.calcHist([image],[1],None,[256],[0,256])
    red_histr = cv2.calcHist([image],[2],None,[256],[0,256])
    
    return equalized_hist, blue_histr, green_histr, red_histr


def getX2results(pathDataset, sourceImageName):
    distanceList = []
    sourceImage = cv2.imread(pathDataset + sourceImageName, 1)
    equalized_hist_source, blue_histr_source, green_histr_source, red_histr_source = getHistograms(sourceImage)

    targetDf = create_df(pathDataset)
    for i in range(len(targetDf)): 
        dfSingle = targetDf.iloc[i]
        targetImage = get_image(dfSingle, pathDataset)
        equalized_hist_target, blue_histr_target, green_histr_target, red_histr_target = getHistograms(targetImage)
        
        distance = x2Distance(equalized_hist_target, equalized_hist_source)
        distanceList.append(distance)
    distanceList.sort()
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

        
pathDataset = "datasets/museum_set_random/"
soruceImageName = "/ima_000000.jpg"
distanceList = getX2results(pathDataset, soruceImageName)
similarityList = getHellingerKernelResult(pathDataset, soruceImageName)
intersectionList = getHistInterseccionResult(pathDataset, soruceImageName)