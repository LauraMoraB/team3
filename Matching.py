# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 17:02:44 2018

@author: gaby1
"""

from ImageFeature import get_cropped_image,get_full_image
import numpy as np
import cv2

def mediatemplate(df, path):
    
    for typeSignal in sorted(df.Type.unique()):
        typeDf = df[df.Type == typeSignal]  
        a=len(typeDf)
        for i in range (a):#(len(typeDf)):
            dfSingle = typeDf.iloc[i]    
            gtimage = get_cropped_image(dfSingle, path)
            imageName = dfSingle['Image'] 
            gtimage= cv2.cvtColor(gtimage, cv2.COLOR_BGR2GRAY)
            #to do a mean we need a float type image 
            gtimage=gtimage.astype(float)
            #save firs image
            if (i==0):
                template = gtimage.copy()
                #divided by number of images
                for k in range(0,template.shape[0]):
                    for j in range(0,template.shape[1]):
                        pixel = template.item(k, j)/a
                        template.itemset(k, j, pixel)
            #media for all images
            if (i>=1):
                sizegt= gtimage.shape[1]*gtimage.shape[0]
                sizetemp= template.shape[1]*template.shape[0]

                #resize to large image
                if(sizetemp>sizegt):
                    dim=(template.shape[1],template.shape[0])    
                    gtimage=cv2.resize(gtimage, dim, interpolation = cv2.INTER_AREA)
                else:
                    dim=(gtimage.shape[1],gtimage.shape[0])    
                    template=cv2.resize(template, dim, interpolation = cv2.INTER_AREA)

                #divided by number of images
                for k in range(0,gtimage.shape[0]):
                    for j in range(0,gtimage.shape[1]):
                        pixel = gtimage.item(k, j)/a
                        gtimage.itemset(k, j, pixel)               
                #add 2 images
                template= cv2.add(gtimage,template)
                cv2.imwrite(path+'cropImage/'+imageName[:-3]+'add.'+'png', template)


def Matching_GRIS(df, path):
     for i in range(len(df)):       
        # Gets images one by one
        dfSingle = df.iloc[i]
        img_rgb  = get_full_image(dfSingle, path)              
        imageName = dfSingle['Image'] 
        img_gray = cv2.cvtColor(img_rgb , cv2.COLOR_BGR2GRAY)
        txtFile = open("TXT/gt."+imageName[:-3]+"txt", "w")
        
        for j in range(1,7):   
            #Load graytemplate
            template = cv2.imread("template/mask.temp"+str(j)+".png",0)           
            w, h = template.shape[::-1]
            #find a distance with image and template
            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.75
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                txtFile.write(str(float(pt[1]))+" "+ str(float(pt[0]))+ " " + str(float(pt[1] + h)) + " " + str(float(pt[0] + w))+"\n")

                cv2.imwrite(path+'cropImage/'+imageName[:-3]+'img_rgb.'+'png', img_rgb)

        txtFile.close()   

