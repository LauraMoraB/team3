# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:48:12 2018

@author: Zaius
"""
import cv2

for i in range(1,9):
    print(i)
    fullImg = cv2.imread("template/temp"+str(i)+".png",1)
    greyRes  = cv2.cvtColor(fullImg, cv2.COLOR_BGR2GRAY)
    greyRes = cv2.bitwise_not(greyRes)
    ret, thresh = cv2.threshold(greyRes, 0, 1, cv2.THRESH_BINARY)
    cv2.imwrite('template/mask.temp'+str(i)+'.png', thresh)


