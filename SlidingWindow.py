import cv2
import numpy as np
from matplotlib import pyplot as plt
from resizeImage import image_resize
import pandas as pd
from OverlapSolution import non_max_suppression_slow

## Method for sliding window
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):     
        for x in range(0, image.shape[1], stepSize):
            # yield the current window         
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
           
            if x+windowSize[0] >= image.shape[1]:
                break
                
        if y+windowSize[1] >= image.shape[0]:
                break

# Method for setting window size
def get_window_size(df):
    # Get window size from Aspect Ratio in the traffic signs
    meanArea = df["Area"].mean()
    meanAspect = df["FormFactor"].mean()
    
    winH = int(np.sqrt(meanArea/meanAspect))
    winW = int(winH*meanAspect)
    
    # Set numbers directly
    #(winW, winH) = (100, 100)
    return (winW, winH)

# Compute Fill Ratio
def compute_fill_ratio(x,y, winW,winH, image):
    imageSegmented = image[y:y+winH,x:x+winW]
    
    fillRatioOnes = np.count_nonzero(imageSegmented)
    sizeMatrix = np.shape(imageSegmented)
    
    fillRatioZeros = sizeMatrix[0]*sizeMatrix[1]
    fillRatio = fillRatioOnes/fillRatioZeros
    
    return fillRatio

def window_detection(image, stepSize, windowSize):
    # init variables
    winW= windowSize[0]
    winH = windowSize[1]
    #fillRatio = []
    candidateWindow= []
    bigger =0

    # loop over the sliding window each image
    for (x, y, window) in sliding_window(resized, stepSize, windowSize=(winW, winH)):
        
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        filling = compute_fill_ratio(x, y, winW, winH, resized)
        
        # Value is now set to 0, but it should be related with: 
        # - the window size
        # - Fill Ratios calculated in task 1
        if filling > 0.0:  
            stepSize=2
            
            # Portion to be analyse deeply 
            maxx=winW*2
            maxy=winH*2
            
            imageSegmented = resized[y:y+maxy,x:x+maxx]      
            
            for (x_in, y_in, window_in) in sliding_window(imageSegmented, stepSize, windowSize=(winW, winH)):
                
                filling = compute_fill_ratio(x_in, y_in, winW, winH, imageSegmented) 
                
                if filling > bigger:
                    bigger=filling
                    startX = x+x_in
                    startY = y+y_in
                    endX = startX+winW
                    endY=startY+winH
                    possibleWindow = [startX, startY, endX, endY]
                
            candidateWindow.append(possibleWindow)
            bigger=0

    # Convert to numpy array for the overlap removal method
    boundingBoxes = np.array(candidateWindow)
    return boundingBoxes

def overlapping_removal(boundingBoxes, overlapThreshold):
    ## Removing Overlapping 
    print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    orig = resized.copy()
    result = resized.copy()

    # JUST TO VISUALIZE
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 255, 255), 2)

    plt.imshow(orig)
    plt.show()


    # Remove Overlapping
    pick = non_max_suppression_slow(boundingBoxes, overlapThreshold)
    print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)) )

    # JUST TO VISUALIZE
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(result, (startX, startY), (endX, endY), (255, 255, 255), 2)
        
    plt.imshow(result)
    plt.show()

    return pick

if __name__ == "__main__":
    # load the image and define the window width and height
    image = cv2.imread("datasets/train/mask/mask.00.000948.png",0)
    (h, w)=image.shape[:2]

    # Mida window
    train = pd.read_csv("trainDataset.csv")
    (winW, winH) = get_window_size(train)

    res = 1

    if res == 1:
        # Image reduced for computation purposes
        resized = image_resize(image, height = int(h/2))
        (h, w)=resized.shape[:2]
        # As the image is resize form computing purposes, the window will be to
        (winW, winH)=(int(winW/2), int(winH/2))
        
    else:
        #Original Size image
        resized = image.copy()

    overlapThreshold=0.3
    stepSize= int(winW*overlapThreshold) # how much overlapp between windows

    allBBoxes = window_detection(stepSize, resized, windowSize=(winW, winH))

    finalBBoxes= overlapping_removal(allBBoxes, overlapThreshold)