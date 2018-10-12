import cv2
import numpy as np
from ImageFeature import get_full_mask, get_full_mask_result
from matplotlib import pyplot as plt
        
def validation(df, path):    

    TrueNeg = 0
    TruePos = 0
    FalsePos = 0
    FalseNeg = 0
    for i in range(len(df)):       
        dfSingle = df.iloc[i]
        maskResult = get_full_mask_result(dfSingle, path)
        plt.figure()
        plt.imshow(maskResult)
        maskResult = cv2.cvtColor(maskResult, cv2.COLOR_BGR2GRAY)
        ret, maskResult = cv2.threshold(maskResult, 0, 10, cv2.THRESH_BINARY)
        maskResult = maskResult*10
        
        maskValidator = get_full_mask(dfSingle, path)
        maskValidator = cv2.cvtColor(maskValidator, cv2.COLOR_BGR2GRAY)
        ret, maskValidator = cv2.threshold(maskValidator, 0, 10, cv2.THRESH_BINARY)
        maskValidator = maskValidator*10
        
        dst = cv2.addWeighted(maskValidator,0.7,maskResult,0.3,0)
        unique, counts = np.unique(dst, return_counts=True)
        dict_FScore = dict(zip(unique, counts))

        for value in unique:
            if value == 0:
                TrueNeg = TrueNeg + dict_FScore[0]
            elif value == 100:
                TruePos = TruePos + dict_FScore[100]
            elif value == 30:
                FalsePos = FalsePos + dict_FScore[30]
            elif value == 70:
                FalseNeg = FalseNeg + dict_FScore[70]
            else:
                print("NOP")
                
            
    print("True Negative :" + str(TrueNeg))
    print("True Positive :" + str(TruePos))
    print("False Negative :" + str(FalseNeg))
    print("False Positive :" + str(FalsePos))

            