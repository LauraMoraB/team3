import cv2
import numpy as np
from ImageFeature import get_cropped_masked_image
from matplotlib import pyplot as plt

def RGB2HSV(image):
    # converts RGB image array into H, S, V, unidimensional ordered list
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    return (image2list(hue), image2list(sat), image2list(val))

def image2list(image):
    # convert image array into unidimensional(pixel) ordered list
    imageL =  image.tolist()
    pxL = []
    for rows in imageL:
        for px in rows:
            pxL.append(px)
    return pxL
            
def get_px_values(dfSingle, path):
    # return hue and sat values for valid px  
    image = get_cropped_masked_image(dfSingle, path)
    hueL, satL, valL = RGB2HSV(image)
    imageL = image2list(image)
    validHue = []
    validSat = []
    pxCount = 0
    for px in imageL:
        if any(px):
            validHue.append(hueL[pxCount])
            validSat.append(satL[pxCount])
        pxCount += 1
    return (validHue, validSat)

def get_color_histogram(df, path):
    # creates color histograms in HSV for each signalType in the df
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
    for typeSignal in sorted(df.Type.unique()):
        typeDf = df[df.Type == typeSignal]    
        hueL = []
        satL = []
        for i in range(len(typeDf)):
            dfSingle = typeDf.iloc[i]     
            (hueSingle, satSingle) = get_px_values(dfSingle, path)
            hueL.extend(hueSingle)
            satL.extend(satSingle)
        plt.hist(hueL, bins = 25, color = colors.pop(0))
        plt.ylabel('f')
        plt.xlabel('hue')
        plt.title('signalType '+typeSignal)
        plt.show()