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

def get_window_size(df):
    """
     Method for setting window size
     to improve
       - min 
        - max
     Adaptar mida live
    """
    # Get window size from Aspect Ratio in the traffic signs
    meanArea = df["Area"].mean()
    meanAspect = df["FormFactor"].mean()
    
    winH = int(np.sqrt(meanArea/meanAspect))
    winW = int(winH*meanAspect)
    
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
    
    candidateWindow= []
    bigger =0

    # loop over the sliding window each image
    for (x, y, window) in sliding_window(image, stepSize, windowSize=(winW, winH)):
        
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        filling = compute_fill_ratio(x, y, winW, winH, image)
        
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
                    possibleWindow = [startY, startX, endY, endX]
                
            candidateWindow.append(possibleWindow)
            
            bigger=0
    # Convert to numpy array for the overlap removal method
    boundingBoxes = np.array(candidateWindow)
    
    return boundingBoxes, candidateWindow

def overlapping_removal(boundingBoxes, overlapThreshold, image):
    
    removal = 1
    
    if removal == 1:
        # Remove Overlapping
        pick = non_max_suppression_slow(boundingBoxes, overlapThreshold)
    else:
        pick = join_bbox(boundingBoxes, overlapThreshold)
        
        
#    # JUST TO VISUALIZE
#        
#    orig = image.copy()
#    result = image.copy()
#    for (startX, startY, endX, endY) in pick:
#        cv2.rectangle(result, (startX, startY), (endX, endY), (255, 255, 255), 2)
    #plt.imshow(result)
    #plt.show()

    return pick


def compute_windows(df, pathToImage, line, res=0): 
    # Get image name
    name = df["Image"].iloc[line]
    split = name.split(".")
    name = split[0]+"."+split[1]
    
    
    # load the image and define the window width and height
    imageRead = df.iloc[line]
   
    image = get_full_mask(imageRead, pathToImage, 1)
   
    
    (h, w)=image.shape[:2]
    (winW, winH) = get_window_size(df)
    
    if res == 1:
        # Image reduced for computation purposes
        image = image_resize(image, height = int(h/2))
        (h, w)=image.shape[:2]
        # As the image is resize form computing purposes, the window will be to
        (winW, winH)=(int(winW/2), int(winH/2))
        
    overlapThreshold=0.3
    stepSize= int(winW*overlapThreshold) # how much overlapp between windows


    severalSizes= []
    for i in range(20,140,20):
        winW=winH=i
        
        stepSize= int(winW*overlapThreshold) # how much overlapp between windows
        allBBoxes, allBBoxes_list = window_detection(image, stepSize, windowSize=(winW, winH))
       
        if allBBoxes_list:
            severalSizes.append(allBBoxes_list[0])
#        else:
#            print("buida")
   
    finalBBoxes = overlapping_removal(np.array(severalSizes), overlapThreshold, image)
    
    if type(finalBBoxes) == np.ndarray:
        listBbox = finalBBoxes.tolist()
    else:
        listBbox = finalBBoxes

    return name,listBbox

def window_main(df, path):
    finalBBoxes =[]
    
    for i in range(df.shape[0]):
         
        name, listb = compute_windows(df, path, i, 0)
            
        finalBBoxes.append((name,listb))  
        
        imageRead = df.iloc[i]
        image = get_full_mask(imageRead, path,1) 
        
        im =cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
            
        for j in range(len(listb)):
            startY = listb[j][0]
            startX = listb[j][1]
            endY = listb[j][2]
            endX = listb[j][3]
            cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 255), 5)
             
        cv2.imwrite(path+'resultWindow/'+df["Mask"].iloc[i], im)
            
    return finalBBoxes
         