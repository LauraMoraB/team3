import cv2
import numpy as np
from resizeImage import image_resize
from OverlapSolution import non_max_suppression_slow
from ImageFeature import get_full_mask

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
                
# to improve
#   - min 
#    - max
# Adaptar mida live
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
def compute_fill_ratio(x,y, winW, winH, image):
    imageSegmented = image[y:y+winH,x:x+winW]
    fillRatioOnes = np.count_nonzero(imageSegmented)
    (imgX, imgY) = np.shape(imageSegmented)
    area = imgX*imgY
    fillRatio = fillRatioOnes/area
    
    return fillRatio

def window_detection(image, stepSize, windowSize):
    # init variables
    winW= windowSize[0]
    winH = windowSize[1]
    #fillRatio = []
    candidateWindow= []
    bigger =0

    # loop over the sliding window each image
    
    for (x, y, window) in sliding_window(image, stepSize, windowSize=(winW, winH)):
        
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        filling = compute_fill_ratio(x, y, winW, winH, image)
        
        # Value is now set to 0, but it should be related with: 
        # - the window size
        # - Fill Ratios calculated in task 1
        if filling > 0.4:  
            stepSize=2
            
            # Portion to be analyse deeply 
            maxx=winW*2
            maxy=winH*2
            
            imageSegmented = image[y:y+maxy,x:x+maxx]      
            
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

def overlapping_removal(boundingBoxes, overlapThreshold, image):
    removal =1
    ## Removing Overlapping 
    #print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
    orig = image.copy()
    result = image.copy()

    # JUST TO VISUALIZE
    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in boundingBoxes:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 255, 255), 2)

    #plt.imshow(orig)
    #plt.show()

    if removal == 1:
        # Remove Overlapping
        pick = non_max_suppression_slow(boundingBoxes, overlapThreshold)
        #print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)) )
    else:
        pick = join_bbox(boundingBoxes, overlapThreshold)
        
        
    # JUST TO VISUALIZE
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(result, (startX, startY), (endX, endY), (255, 255, 255), 2)
        
    #plt.imshow(result)
    #plt.show()

    return pick


def compute_windows(dfTrain, pathToImage, line, res=0):    
    # load the image and define the window width and height
    imageRead = dfTrain.iloc[line]
    #image = cv2.imread('datasets/split/train/mask/mask.00.005892.png',0)
    image = get_full_mask(imageRead, pathToImage)
    
    #image = cv2.imread(pathToImage+imageRead,0)
    
    (h, w)=image.shape[:2]
    (winW, winH) = get_window_size(dfTrain)
    
    if res == 1:
        # Image reduced for computation purposes
        image = image_resize(image, height = int(h/2))
        (h, w)=image.shape[:2]
        # As the image is resize form computing purposes, the window will be to
        (winW, winH)=(int(winW/2), int(winH/2))
        
    
    #plt.imshow(image, cmap='gray')
    #plt.show()
    
    
    overlapThreshold=0.3
    stepSize= int(winW*overlapThreshold) # how much overlapp between windows

    allBBoxes = window_detection(image, stepSize, windowSize=(winW, winH))

    finalBBoxes = overlapping_removal(allBBoxes, overlapThreshold,image)
    
    return allBBoxes,finalBBoxes


def window_main(df, trainSplitPath):
    for i in range(0,1):#(dfTrain.shape[0]):
        
        imageRead = dfTrain.iloc[i]
        image = get_full_mask(imageRead, trainSplitPath)  
        
        beforeSel, selection = compute_windows(dfTrain, trainSplitPath, i,0)
        
        im =cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
        
        for j in range(len(selection)):
            startX = selection[j][0]
            startY = selection[j][1]
            endX = selection[j][2]
            endY = selection[j][3]
            cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 255), 5)
         
        #cv2.imwrite(trainSplitPath+'resultWindow/'+dfTrain["Mask"].iloc[i], im)