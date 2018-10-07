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

def pixelescolorDetectionLuv(imagen, colorType,signalType, name):
    kernel = np.ones((6,6),np.uint8)
    if colorType == "red":
        rojo_bajos1 = np.array([13.858842702172808,45.57394612919739,9.828104239782657], dtype=np.uint8)
        rojo_altos1 = np.array([53.23288178584245,175.05303573649485,37.750505032665004], dtype=np.uint8)

        print("rojo_altos2")
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)
    elif colorType == "blue":
        blue_1 = np.array([0.8467307788802485,-0.2463938064858469,-3.4170166322299997], dtype=np.uint8)
        blue_2 = np.array([32.302586667249486, -9.39986768735168, -130.35840748816472], dtype=np.uint8)
   
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mascara_blue)
           
    return mask


def pixelescolorDetectionHLS(imagen, colorType,signalType, name):
#    kernel = np.ones((6,6),np.uint8)
    Lchannel = imagen[:,:,1]
    if colorType == "red":
#        rojo_bajos1 = np.array([0,100,50], dtype=np.uint8)
#        rojo_altos1 = np.array([0,100,127], dtype=np.uint8)

        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
#        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        mascara_rojo1 = cv2.inRange(Lchannel, 50, 100)

        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
#        mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        mask=mascara_rojo1
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)
    elif colorType == "blue":
        blue_1 = cv2.inRange(Lchannel, 10, 150)
#       mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)

#        blue_1 = np.array([240,255,10], dtype=np.uint8)
#        blue_2 = np.array([240, 255, 255], dtype=np.uint8)
#   
#        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
#        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask  = blue_1
    plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mask)
           
    return mask
def pixelescolorDetectionLAB(imagen, colorType,signalType, name):
    kernel = np.ones((6,6),np.uint8)
    if colorType == "red":
        rojo_bajos1 = np.array([16.373412184010128,37.45924812443255,25.407550256563038], dtype=np.uint8)
        rojo_altos1 = np.array([53.23288178584245,80.10930952982204,67.22006831026425], dtype=np.uint8)

        print("rojo_altos2")
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)
    elif colorType == "blue":
        blue_1 = np.array([2.0802037779242646,14.617714357683415,-29.445881245339127], dtype=np.uint8)
        blue_2 = np.array([32.302586667249486, 79.19666178930935, -107.86368104495168], dtype=np.uint8)
   
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mascara_blue)
           
    return mask
###############################################################################
def pixelescolorDetectionYCrCb(imagen, colorType,signalType, name):
 #NO ENCUENTRO CAMBIO AUN    
    kernel = np.ones((6,6),np.uint8)
    if colorType == "red":
        rojo_bajos1 = np.array([0,100,50], dtype=np.uint8)
        rojo_altos1 = np.array([0,100,127], dtype=np.uint8)

        print("rojo_altos2")
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)
    elif colorType == "blue":
        blue_1 = np.array([240,255,10], dtype=np.uint8)
        blue_2 = np.array([240, 255, 255], dtype=np.uint8)
   
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mascara_blue)
        return mask
def pixelescolorDetectionXYZ(imagen, colorType,signalType, name):
    kernel = np.ones((6,6),np.uint8)
    if colorType == "red":
        #ERRORES DE CONVERSION 1,1,1.93?????????????????????????
        #ERRORES DE CONVERSION 1,1,0.19732654486003556
        rojo_bajos1 = np.array([4,216449072553298,2.173659245453034,0.19732654486003556], dtype=np.uint8)
        rojo_altos1 = np.array([41.24,21.26,1.93], dtype=np.uint8)

        print("rojo_altos2")
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_rojo1 = cv2.inRange(imagen, rojo_bajos1, rojo_altos1)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mask_red.jpg', mask)
    elif colorType == "blue":
        blue_1 = np.array([0.2343437337762229,0.09373749351048916,1.2340372241235449], dtype=np.uint8)
#        blue_1 = np.array([0.2343437337762229,0.09373749351048916,1], dtype=np.uint8)
        blue_2 = np.array([18.05, 7.22, 95.05], dtype=np.uint8)
 #       blue_2 = np.array([1,1, 1], dtype=np.uint8)
  
        #Detectar los pixeles de la imagen que esten dentro del rango de rojos
        mascara_blue  = cv2.inRange(imagen, blue_1, blue_2)
        
        #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
        mask  = cv2.morphologyEx(mascara_blue, cv2.MORPH_CLOSE, kernel)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'mascara_blue.jpg', mascara_blue)
    return mask           
def pixelescolorDetectionHSV(imagen, colorType,signalType, name):
    #Filtrar el ruido aplicando un OPEN seguido de un CLOSE
    #plt.imsave("./Resultados/"+signalType+"/"+name+'imagenentradapixelcolorDetection.jpg', imagen)        
    kernel = np.ones((6,6),np.uint8)

    if colorType == "red":
        rojo_bajos1 = np.array([0,50,60], dtype=np.uint8)
        rojo_altos1 = np.array([20, 255, 255], dtype=np.uint8)
        rojo_bajos2 = np.array([300,75,60], dtype=np.uint8)
        rojo_altos2 = np.array([360, 255, 255], dtype=np.uint8)
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
        blue_1 = np.array([100,50,40], dtype=np.uint8)
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

    plt.imsave("./Resultados/"+signalType+"/"+name+'changeSpaceColorRGB.jpg', imagen)
    if spaceType == "HSV":

        hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
        #plt.imsave("./Resultados/"+signalType+"/"+name+'HSV.jpg', hsv)
        return hsv
    elif spaceType == "HLS":
#        hls = cv2.cvtColor(imagen, cv2.COLOR_RGB2HLS )
        hls = cv2.cvtColor(imagen, cv2.COLOR_BGR2HLS )

        plt.imsave("./Resultados/"+signalType+"/"+name+'changeSpaceColorHLS.jpg', hls)
        return hls
    
    elif spaceType == "LAB":   
        lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
        return lab

    elif spaceType == "YCrCb":   
        ycrcb = cv2.cvtColor(imagen, cv2.COLOR_RGB2YCrCb)
        return ycrcb

    elif spaceType == "XYZ":   
        xyz = cv2.cvtColor(imagen, cv2.COLOR_RGB2XYZ)
        return xyz
    elif spaceType == "Luv":   
        Luv = cv2.cvtColor(imagen, cv2.COLOR_RGB2Luv )
        return Luv
    else:
        return imagen
def computeColor(image_dict, spaceType, colorType):

    color_dict = {}   
    for signalType in image_dict:
#        chRGB_list = []
        SpaceColors_list = []
        for channelGrid in image_dict[signalType]:
#           imageRGB = cv2.cvtColor(channelGrid.finalGrid, cv2.COLOR_BGR2RGB)
#           plt.imsave("./Imagen recortada "+channelGrid.name+'.jpg', channelGrid.finalGrid)
            #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'.jpg', imageRGB)
#            imghsv=changeSpaceColor(imageRGB, spaceType, signalType,channelGrid.name)
            imghsv=changeSpaceColor(channelGrid.finalGrid, spaceType, signalType,channelGrid.name)
            #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'imghsv.jpg', imghsv)

            if spaceType=="HSV":    
                #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'imghsv.jpg', imghsv)
                maskcolor=pixelescolorDetectionHSV(imghsv,colorType,signalType,channelGrid.name)
                #plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'maskcolor.jpg', maskcolor)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)

            elif spaceType=="HLS":    
                plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'ANTESpixelescolorDetectionHLS.jpg', imghsv)
                maskcolor=pixelescolorDetectionHLS(imghsv,colorType,signalType,channelGrid.name)
                plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+'DESPESpixelescolorDetectionHLS.jpg', maskcolor)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
                plt.imsave("./Resultados/"+signalType+"/"+channelGrid.name+spaceType+'resultSpaceHLS.jpg', resultSpace)       

                # Bitwise-AND mask and original image
            elif spaceType == "LAB":   
                maskcolor=pixelescolorDetectionLAB(imghsv,colorType,signalType,channelGrid.name)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)

            elif spaceType == "YCrCb":   
                maskcolor=pixelescolorDetectionYCrCb(imghsv,colorType,signalType,channelGrid.name)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
    
            elif spaceType == "XYZ":   
                maskcolor=pixelescolorDetectionXYZ(imghsv,colorType,signalType,channelGrid.name)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
        
            elif spaceType == "Luv":   
                maskcolor=pixelescolorDetectionLuv(imghsv,colorType,signalType,channelGrid.name)
                resultSpace = cv2.bitwise_and(imghsv,imghsv, mask= maskcolor)
                        
    #            chRHSV_list.append(res)       
            
            SpaceColors_list.append(resultSpace)
#        channel_M_dict[signalType] = channel_list2
        color_dict[signalType] = SpaceColors_list
    return (color_dict)
     



if __name__ == '__main__':
    imgType = 'D' 
    spaceType = 'HLS'  
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
    

   