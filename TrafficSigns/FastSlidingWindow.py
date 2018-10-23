import cv2
import numpy as np
from ImageFeature import get_full_mask, get_full_mask_result, get_full_image, save_text_file,get_full_mask_window_result
import os

def fast_sw(df, path, dfStats):
#    start_time = time.clock()
    # Create constraints from DS study
    ROWS_MIN = min(dfStats['YMin'].tolist())
    COLS_MIN = min(dfStats['XMin'].tolist())
    FILL_RATIO_MIN = min(dfStats['FillRatioMin'].tolist())
    # dictionary to store BBlist per image
    dsListBB=[]
    for i in range(len(df)):
        dfSingle = df.iloc[i]
        mask = get_full_mask_window_result(dfSingle, path+"resultMask/")
        # First BB candidate --> full image      
        (imgRows, imgCols) = np.shape(mask)
        BB = np.array([[0,0],[imgRows,imgCols]])
        # iterate!
        listBB = evaluate_image_wrap(mask, FILL_RATIO_MIN, ROWS_MIN, COLS_MIN, BB)
        # joinb candidate bbs
        joinBB = join_bbs(listBB.copy())
        # rejoinv posible new intersections
        finalBB = join_bbs(joinBB.copy())
        # safe and boxes polotting!       
        to_list(finalBB)
        dsListBB.append((dfSingle['Image'][0:-4],to_list(finalBB)))  
        
       
        im = cv2.cvtColor(mask.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
        
        pathSave="datasets/test/gtResult/PKL/FAST_PICKEL_MASK/"
        mask_name=dfSingle['Image'][0:-4]+".png"
        
        
        listb = to_list(finalBB)
    
        mask2 = np.zeros(im.shape, np.uint8)
        
        if len(listb)>0:
            for i in range(0, len(listb)):
                #print (mask_name, listb[i][1],listb[i][3],listb[i][0],listb[i][2])
                mask2[listb[i][1]:listb[i][3],listb[i][0]:listb[i][2]] = im[listb[i][1]:listb[i][3],listb[i][0]:listb[i][2]]
#                from matplotlib import pyplot as plt
#                plt.imshow(mask2)
#                plt.show()
#        #Save Images
        if not os.path.exists(pathSave):
            os.makedirs(pathSave)
            
        from matplotlib import pyplot as plt
        
        cv2.imwrite(pathSave+mask_name, mask2)
        
#        # save visual result for testing purposes
#        maskbw =cv2.cvtColor(mask.astype('uint8') * 255, cv2.COLOR_GRAY2BGR)        
#        for j in range(len(finalBB)):
#            BB = finalBB[j].tolist()
#            cv2.rectangle( maskbw, (BB[0][1],BB[0][0]), (BB[1][1],BB[1][0]), (0, 255, 255), 5)
#        subPath = "resultWindows/fast/"
#        totalPath = path + subPath
#        if not os.path.exists(totalPath):
#            os.makedirs(totalPath)
#        cv2.imwrite(totalPath+dfSingle['Image'], maskbw)
         
    return dsListBB

def to_list(list_in):
    bb_list = []
    for bb in list_in:
        bb_list.append([bb[0][0],bb[0][1], bb[1][0],bb[1][1]])
    return bb_list
        
def evaluate_image_wrap(img, fillRatioMin, rowsMin, colsMin, BB):   
    listBB, currentBB, status = evaluate_image(img.copy(), fillRatioMin, rowsMin, colsMin, BB.copy(), BB.copy(), listBB=[])
    return listBB
        
def evaluate_image(img, fillRatioMin, rowsMin, colsMin, currentBB, oldBB, listBB):
    (imgRows, imgCols) = np.shape(img)
    imgOnes = np.count_nonzero(img)  
    imgArea = imgRows*imgCols
    imgFillRatio = imgOnes/imgArea
    
    if(imgRows < rowsMin or imgCols < colsMin):
        # Windows already too small
        return listBB, oldBB, False
    elif(imgFillRatio > fillRatioMin):
        # Candidate found
        listBB.append(currentBB.copy())
        return listBB, oldBB, False   
    else:
        # FillRatio too small, need to zoom in and evaluate again!
        halfRows = int(imgRows*.45)
        halfCols = int(imgCols*.45)
        # top-left quadrant
        oldBB = currentBB.copy()
        currentBB += np.array([[0, 0],[-halfRows, -halfCols]])       
        subImg = img[:-(halfRows+1) , :-(halfCols+1)]
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, rowsMin, colsMin, currentBB, oldBB, listBB)
        # top-right quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[0, halfCols+1],[-halfRows, 0]])       
        subImg = img[:-(halfRows+1) , halfCols:]
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, rowsMin, colsMin, currentBB, oldBB, listBB)
        # bottom-left quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[halfRows+1, 0],[0,-halfCols]])    
        subImg = img[halfRows:, :-(halfCols+1)]
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, rowsMin, colsMin, currentBB, oldBB, listBB)
        # bottom-right quadrant
        if(status):
            currentBB = oldBB.copy()
        else:
            oldBB = currentBB.copy()
        currentBB += np.array([[halfRows+1, halfCols+1],[0, 0]])     
        subImg = img[halfRows:,  halfCols:]
        listBB, currentBB, status = evaluate_image(subImg, fillRatioMin, rowsMin, colsMin, currentBB, oldBB, listBB)
        return listBB, currentBB, True
    
    return listBB, oldBB, False

def join_bbs(listBB):
    jointBB = []
    sortedBB = sorted(listBB, key=lambda x:x[-1][-1])
    while(len(sortedBB)>0):
        jointBB.append(sortedBB.pop())
        for bb in reversed(sortedBB):
            if(overlap(bb, jointBB[-1])):
                jointBB[-1] = max_bb(bb, jointBB[-1])  
                sortedBB.pop()
    return jointBB
                
def overlap(bb1, bb2):
    dr = min(bb1[1][0], bb2[1][0]) - max(bb1[0][0], bb2[0][0])
    dc = min(bb1[1][1], bb2[1][1]) - max(bb1[0][1], bb2[0][1])
    if(dc>=0 and dr>=0):
        return dc*dr
    else :
        return 0

def max_bb(bb1, bb2):
    (rmin, cmin) = (min(bb1[0][0], bb2[0][0]), min(bb1[0][1], bb2[0][1]))
    (rmax, cmax) = (max(bb1[1][0], bb2[1][0]), max(bb1[1][1], bb2[1][1]))
    return np.array([[rmin,cmin],[rmax,cmax]])
        
#def show_bb(dfSingle, path, bb = []):
#    img = get_full_image(dfSingle, path)
#    if (len(bb)):
#        img = img[bb[0][0]:bb[1][0],bb[0][1]:bb[1][1]]
#    imgrgb =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)        
#    plt.imshow(imgrgb)
#    plt.show()       
#        
        
        
        
        
        
        