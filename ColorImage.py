# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 09:31:44 2018

@author: gaby1
"""
import cv2
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from ImageFeature import getGridOfImage 


def pixelescolorDetection(imagen, colorType,signalType, name):
    #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
    #plt.imsave("./Resultados/"+signalType+"/"+name+'imagenentradapixelcolorDetection.jpg', imagen)        
    kernel = np.ones((6,6),np.uint8)
    print(colorType)
    print(signalType)
    print(name)

    if colorType == "red":
        rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
        rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
        rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
        rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)
        print("rojo_altos2")
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        mascara_rojo2 = cv2.inRange(imagen, rojo_bajos2, rojo_altos2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_rojo1 = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        mascara_rojo2 = cv2.morphologyEx(mascara_rojo2, cv2.MORPH_OPEN, kernel)
        #Unir las dos mascaras con el comando cv2.add() del color rojo
        mask = cv2.add(mascara_rojo1, mascara_rojo2)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)        

    elif colorType == "black":
        black_1 = np.array([0,0,0], dtype=np.uint8)
        black_2 = np.array([180, 255, 30], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_black = cv2.inRange(imagen, black_1, black_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_black = cv2.morphologyEx(mascara_black, cv2.MORPH_CLOSE, kernel)
        #Mascara de pixeles negros
        mask =  mascara_black
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_black.jpg', mask)        
    elif colorType == "white":
        white_1 = np.array([0,0,200], dtype=np.uint8)
        white_2 = np.array([180, 255, 255], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_white = cv2.inRange(imagen, white_1, white_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_white = cv2.morphologyEx(mascara_white, cv2.MORPH_CLOSE, kernel)
        #Mascara de pixeles blancos
        mask =  mascara_white
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_white.jpg', mask)        

        
    elif colorType == "blue":
        blue_1 = np.array([100,150,0], dtype=np.uint8)
        blue_2 = np.array([140, 255, 255], dtype=np.uint8)
        
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mascara_blue  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mascara_blue)

        #Mascara de pixeles blancos
        mask =  mascara_blue  
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_blue.jpg', mask)        

    else:
        print("Any color signal mask !")        
    return mask

def changeSpaceColor(imagen, spaceType, signalType,name):
#SOLO CAMBIO DE ESPACIO DE COLO LA IMAGEN ORIGINAL RECORTADA

    plt.imsave("./Resultados/"+signalType+"/"+name+'RGB.jpg', imagen)
    if spaceType == "HSV":

        hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'HSV.jpg', hsv)
        return hsv
    else:
        return imagen
def computeColor(image_dict, spaceType, colorType):

    color_dict = {}   
    for signalType in image_dict:
#        chRGB_list = []
        chHSV_list = []
        for channelGrid in image_dict[signalType]:
            imageRGB = cv2.cvtColor(channelGrid.finalGrid, cv2.COLOR_BGR2RGB)
#           plt.imsave("./Imagen recortada "+channelGrid.name+'.jpg', channelGrid.finalGrid)
            #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'.jpg', imageRGB)
            imghsv=changeSpaceColor(imageRGB, spaceType, signalType,channelGrid.name)
            #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'imghsv.jpg', imghsv)

            if spaceType=="HSV":    
                #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'imghsv.jpg', imghsv)
                maskcolor=pixelescolorDetection(imghsv,colorType,signalType,channelGrid.name)
                #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'maskcolor.jpg', maskcolor)

                # Bitwise-AND mask and original image
                restimghsv = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
                plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'restimghsv.jpg', restimghsv)

    #            chRHSV_list.append(res)       
            
            chHSV_list.append(imghsv)
#        channel_M_dict[signalType] = channel_list2
        color_dict[signalType] = chHSV_list
    return (color_dict)
     



if __name__ == '__main__':
    imgType = 'D' 
    spaceType = 'HSV'  
#    spaceTypeRGB = 'RGB' 
    colorType = 'blue'         
    try:
        (color_dict) = computeColor(image_dict, spaceType, colorType) 
        #(color_dict) = computeColor(image_dict, spaceTypeRGB, colorType) 
    except NameError:
        image_dict = getGridOfImage()
        (color_dict) = computeColor(image_dict, spaceType, colorType) 
    


    plt.imshow(color_dict[imgType][2])
    plt.title('signalType '+imgType)
    plt.show()

#    plt.imshow(color_dict[imgType][1])
#    plt.title('signalType '+imgType+' spaceType'+spaceTypeRGB)
#    plt.show()
    

   