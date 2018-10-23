import numpy as np
import cv2
from createDataframe import load_annotations
import os

def get_full_image(dfSingle, path):
    return cv2.imread(path + dfSingle['Image'],1)

def get_cropped_image(dfSingle, path):
    image = get_full_image(dfSingle, path)
    return image[int(dfSingle["UpLeft(Y)"]):int(dfSingle["DownRight(Y)"]), int(dfSingle["UpLeft(X)"]):int(dfSingle["DownRight(X)"])]

def get_full_mask(dfSingle, path):
    return cv2.imread(path+'mask/' + dfSingle['Mask'], 0)

def get_full_mask_window_result(dfSingle, path):
    split = dfSingle["Image"].split(".")
    name_mask = "mask."+split[0]+"."+split[1]+".png"
    return cv2.imread(path+name_mask, 0)

def get_full_mask_result(dfSingle, path, maskType):
    return cv2.imread(path+'resultMask/'+maskType+'/' + dfSingle['Mask'], 0)

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

def save_gt(pathToSave, pathToget, gtfile):
    annotations = load_annotations(pathToget+gtfile)
    
    with open(pathToSave+gtfile, "w") as f:
        for i,element in enumerate(annotations[0]):   
            if i != 4:
                f.write(format(element, '.8f'))
                f.write(" ")
            else:
                f.write(element)
                
def save_text_file(pathToSave, elements, name, method, typeW):
    name = "gt."+name+".txt"
    path = pathToSave+typeW+"/"
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+name, "a") as f:
        if elements:
            for i,element in enumerate(elements[0]):
                if i != 4:
                    f.write(format(element, '.8f'))
                    f.write(" ")
                else:
                    f.write(element+"\n")
            f.write("\n")
        else:
            f.write("")

def get_mask_aspect(dfSingle, path): 
    crop = get_cropped_mask(dfSingle, path)  
    (imgX, imgY) = np.shape(crop)
    imgOnes = np.count_nonzero(crop)    
    imgFillRatio = imgOnes/(imgX*imgY)
    
    if(imgX < imgY):    
        imgFormFactor = abs(imgX/imgY)
    else:
        imgFormFactor = abs(imgY/imgX)

    return imgFillRatio, imgFormFactor, imgX, imgY

def get_ground_truth(df, path):
    fillRatioL = []
    formFactorL = []
    areaL = []
    xL = []
    yL = []
    
    for i in range(len(df)):       
        fillRatio, formFactor, x, y = get_mask_aspect(df.iloc[i], path)    
        xL.append(x)
        yL.append(y)
        areaL.append(x*y)
        fillRatioL.append(fillRatio)
        formFactorL.append(formFactor)
   
    df["FillRatio"]=fillRatioL
    df["FormFactor"]=formFactorL
    df["Area"]=areaL
    df["X"]=xL
    df["Y"]=yL
        
    return df
