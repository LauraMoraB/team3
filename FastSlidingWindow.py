import cv2
import numpy as np
from resizeImage import image_resize
from OverlapSolution import non_max_suppression_slow
from ImageFeature import get_full_mask
from matplotlib import pyplot as plt

def fast_sw(df, path, dfStats):
    AREA_MIN = min(dfStats['AreaMin'].tolist())
    FILL_RATIO_MIN = min(dfStats['FillRatioMin'].tolist())
    windows_dic={}
    for i in range(len(df)):
        dfSingle = df.iloc[i]
        mask = get_full_mask(dfSingle, path)
        plt.imshow(mask)
        plt.show()
        (imgRows, imgCols) = np.shape(mask)
        BB = np.array([[0,0],[imgRows,imgCols]])
        listBB, currentBB, status = evaluate_image(mask, FILL_RATIO_MIN, AREA_MIN, BB.copy(), BB.copy(), listBB=[])
        windows_dic[dfSingle['Image']]=listBB.copy()
    return windows_dic
        
        
def evaluate_image(img, fillRatioMin, areaMin, currentBB, oldBB, listBB):
    (imgRows, imgCols) = np.shape(img)
    imgOnes = np.count_nonzero(img)  
    imgArea = imgRows*imgCols
    imgFillRatio = imgOnes/imgArea
    
    if(imgArea < areaMin):
        # Considered windows already too small
#        print("No candidate found, final searchBB at", tempBB)
#        print('SMALL: Current',currentBB,'Older',oldBB)
        return listBB, oldBB, False
    elif(imgFillRatio > fillRatioMin):
        # Candidate found
        plt.imshow(img)
        plt.show()
        listBB.append(currentBB.copy())
#        print('FOUND: Current',currentBB,'Older',oldBB)
        return listBB, oldBB, False   
    else:
        # FillRatio too small, need to zoom in and evaluate again!
        halfRows = int(imgRows*.5)
        halfCols = int(imgCols*.5)
        # top-left quadrant
        oldBB = currentBB.copy()
        currentBB += np.array([[0, 0],[-halfRows, -halfCols]])       
        subImg = img[:halfRows , :halfCols]
#        print('TOP LEFT: Current',currentBB,'Older',oldBB)
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, areaMin, currentBB, oldBB, listBB)
        # top-right quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[0, halfCols+1],[-halfRows, 0]])       
        subImg = img[:halfRows , halfCols+1:]
#        print('TOP RIGHT: Current',currentBB,'Older',oldBB)
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, areaMin, currentBB, oldBB, listBB)
        # bottom-left quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[halfRows+1, 0],[0,-halfCols]])    
        subImg = img[halfRows+1:, :halfCols]
#        print('BOT LEFT: Current',currentBB,'Older',oldBB)
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, areaMin, currentBB, oldBB, listBB)
        # bottom-right quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[halfRows+1, halfCols+1],[0, 0]])     
        subImg = img[halfRows+1:,  halfCols:]
#        print('BOT RIGHT: Current',currentBB,'Older',oldBB)
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, areaMin, currentBB, oldBB, listBB)
        return listBB, currentBB, True
    
    return listBB, oldBB, False
        
    
        
