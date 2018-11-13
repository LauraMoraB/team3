import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def groupLines(lines):
    groupedLines = []
    if lines is not None:
        for pack in lines:
            rho = pack[0,0]
            theta = pack[0,1]
            if len(groupedLines)>0:
                controlerModifier = False
                for group in groupedLines:
                    if(theta+0.1>group[2] and theta-1<group[2]):
                        controlerModifier = True
                        if (rho<group[0]):
                            group[0]=rho
                        elif ( rho> group[1]):
                            group[1]=rho
                            
                if (controlerModifier == False):
                    groupedLines.append([rho, rho, theta])
            else:
                groupedLines.append([rho, rho, theta])
    return groupedLines

def intersection(line1, line2):
    points = []
    theta1 = line1[2]
    theta2 = line2[2]
    for i in range(0,2):
        for j in range(0,2):
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[line1[i]], [line2[j]]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            points.append([x0,y0])
    return points


def houghTrasnformPaired(img):
    edges = cv2.Canny(img,100,200, None, 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 125, None, 0, 0)
    groupedLines = groupLines(lines)
    if len(groupedLines)>0:
#        for pack in groupedLines:
        pack1 = groupedLines[0]
        pack2 = groupedLines[1]
        
        points = intersection(pack1, pack2)
        for point in points:
            img = cv2.circle(img,(point[0],point[1]), 10, (0,0,255), -1)
        for j in range(0,2):
            pack = groupedLines[j]
            theta = pack[2]
            a = np.cos(theta)
            b = np.sin(theta)
            for i in range(0,2):
                x0 = a*pack[i]
                y0 = b*pack[i]
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    plt.imshow(img)
    plt.show()


def houghTrasnformGrouped(img):
    edges = cv2.Canny(img,100,200, None, 3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 125, None, 0, 0)
    groupedLines = groupLines(lines)
    if len(groupedLines)>0:
#        for pack in groupedLines:
        for j in range(0,2):
            pack = groupedLines[j]
            theta = pack[2]
            a = np.cos(theta)
            b = np.sin(theta)
            for i in range(0,2):
                x0 = a*pack[i]
                y0 = b*pack[i]
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    plt.imshow(img)
    plt.show()




def houghTrasnform(img):
    edges = cv2.Canny(img,100,200, None, 3)
#    edges = cv2.dilate(edges, np.ones((5,5),np.uint8) ,iterations = 1)
        
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100, None, 0, 0)
    
#    lines_sort = sorted(lines, key=lambda a_entry: a_entry[..., 1], reverse=True)
    counterX = 0
    thetaX = 0
    rhoX = 0
    counterY = 0
    thetaY = 0
    rhoY = 0
    if lines is not None:
        for pack in lines:
            rho = pack[0,0]
            theta = pack[0,1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            deltaX = x1 - x2
            deltaY = y1 - y2
            angle = np.arctan2(deltaY, deltaX)*180 / np.pi
            if (angle > -135 and angle < 135):
                if(abs(theta - thetaX)+abs(rho - rhoX)>50):
                    counterX +=  1
                    if (counterX<=2):
                        thetaX = theta
                        rhoX = rho
                        img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            else :
                if(abs(theta - thetaY)+abs(rho - rhoY)>50):
                    counterY +=  1
                    if (counterY<=2):
                        thetaY = theta
                        rhoY = rho
                        img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    plt.imshow(img)
    plt.show()
    
def probabilisticHoughTrasnform(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    plt.imshow(img)
    plt.show()