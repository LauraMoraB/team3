import cv2
import numpy as np
from ImageFeature import get_full_image, get_full_masked_image, get_cropped_masked_image
from matplotlib import pyplot as plt


#
def RGB2HSV_list(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    return (hue.tolist(), sat.tolist(), val.tolist())



def new_thresholds(df, path):
    image = get_cropped_masked_image(df.iloc[0], path)
    imageL = image.tolist()
    hueL, satL, valL = RGB2HSV_list(image)
    pxL = []
    for px in imageL:
        if(sum(px) >0):
            pxL.append([hueL, satL])
    return pxL
    





def compute_histogram_type(signal_type, image_dict):
    hueL=[]
    satL=[]
    valL=[]
    for i in range((len(image_dict[signal_type]))):
        img = image_dict[signal_type][i]
        testImg = img.finalGrid

        hsv = cv2.cvtColor(testImg, cv2.COLOR_BGR2HSV)

        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        hueL.append(np.ndarray.flatten(hue))
        satL.append(np.ndarray.flatten(sat))
        valL.append(np.ndarray.flatten(val))
        
    return hueL, satL, valL

def pixeles_color_detection_HSV(imagen, signalType, name):
    
    mask_red = pixeles_color_Detection_HSV_individual(imagen, "red", signalType, name)
    mask_blue = pixeles_color_Detection_HSV_individual(imagen, "blue", signalType, name)
    
    #multipl = cv2.bitwise_and(mask_red, mask_red, mask=mask_blue)
    
    mask = cv2.add(mask_red, mask_blue)
    
    #mask = cv2.subtract(suma, multipl)
    
    return mask

def pixeles_color_Detection_HSV_individual(imagen, colorType,signalType, name):
    #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
    kernel = np.ones((6,6),np.uint8)
    
    if colorType == "red":
        rojo_bajos1 = np.array([0,50,60], dtype=np.uint8)
        rojo_altos1 = np.array([20, 255, 255], dtype=np.uint8)
        rojo_bajos2 = np.array([300,75,60], dtype=np.uint8)
        rojo_altos2 = np.array([360, 255, 255], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        mascara_rojo2 = cv2.inRange(imagen, rojo_bajos2, rojo_altos2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_rojo1 = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        mascara_rojo2 = cv2.morphologyEx(mascara_rojo2, cv2.MORPH_OPEN, kernel)
        #Unir las dos mascaras con el comando cv2.add() del color rojo
        mask = cv2.add(mascara_rojo1, mascara_rojo2)

    elif colorType == "black":
        black_1 = np.array([0,0,0], dtype=np.uint8)
        black_2 = np.array([180, 255, 30], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_black = cv2.inRange(imagen, black_1, black_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_black = cv2.morphologyEx(mascara_black, cv2.MORPH_CLOSE, kernel)
        #Mascara de pixeles negros
        mask =  mascara_black

    elif colorType == "white":
        white_1 = np.array([0,0,200], dtype=np.uint8)
        white_2 = np.array([180, 255, 255], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_white = cv2.inRange(imagen, white_1, white_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_white = cv2.morphologyEx(mascara_white, cv2.MORPH_CLOSE, kernel)
        #Mascara de pixeles blancos
        mask =  mascara_white

        
    elif colorType == "blue":
        blue_1 = np.array([100,50,40], dtype=np.uint8)
        blue_2 = np.array([140, 255, 255], dtype=np.uint8)
   
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_blue  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)

        #Mascara de pixeles azules
        mask =  mascara_blue
        
    else:
        print("Any color signal mask !")        
    return mask

    
def compute_color(image_dict, spaceType, tipoFiltro):

    color_dict = {}  

    
    for signalType in image_dict:
        MaskColors_list = []

        for channelGrid in image_dict[signalType]:
            
            imageRGB = cv2.cvtColor(channelGrid.completeImg, cv2.COLOR_BGR2RGB)
            
            imghsv=change_space_color(imageRGB, spaceType, signalType,channelGrid.name)

            if spaceType=="HSV":
                
                if tipoFiltro == "mix":
                    maskcolor=pixeles_color_detection_HSV(imghsv,signalType,channelGrid.name)
                else:    
                    maskcolor=pixeles_color_Detection_HSV_individual(imghsv,colorType,signalType,channelGrid.name)
                           
            MaskColors_list.append([maskcolor,channelGrid.name])
        
        color_dict[signalType] = MaskColors_list
        
    return color_dict
    
    

   