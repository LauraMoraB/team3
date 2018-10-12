import numpy as np
import cv2

def get_full_image(dfSingle, path):
    return cv2.imread(path + dfSingle['Image'],1)

def get_cropped_image(dfSingle, path):
    image = get_full_image(dfSingle, path)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def get_full_mask(dfSingle, path):
    return cv2.imread(path+'mask/' + dfSingle['Mask'], 0)

def get_full_mask_result(dfSingle, path):
    return cv2.imread(path+'resultMask/' + dfSingle['Mask'], 0)

def get_cropped_mask(dfSingle, path):
    image = get_full_mask(dfSingle, path)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def get_full_masked_image(dfSingle, path):
    image = get_full_image(dfSingle, path)
    mask = get_full_mask(dfSingle, path)
    return cv2.bitwise_and(image,image,mask = mask)

def get_cropped_masked_image(dfSingle, path):
    image = get_cropped_image(dfSingle, path)
    mask = get_cropped_mask(dfSingle, path)
    return cv2.bitwise_and(image,image,mask = mask)

def get_mask_aspect(dfSingle, path): 
    crop = get_cropped_mask(dfSingle, path)  
    (imgX, imgY) = np.shape(crop)
    imgOnes = np.count_nonzero(crop)    
    imgArea = imgX*imgY
    imgFillRatio = imgOnes/imgArea
    if(imgX < imgY):    
        imgFormFactor = abs(imgX/imgY)
    else:
        imgFormFactor = abs(imgY/imgX)

    return imgFillRatio, imgFormFactor, imgArea

def get_ground_truth(df, path):
    fillRatioL = []
    formFactorL = []
    areaL = []
    
    for i in range(len(df)):       
        fillRatio, formFactor, area = get_mask_aspect(df.iloc[i], path)    
        areaL.append(area)
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
       
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
    df["Area"]=areaL
        
    return df

