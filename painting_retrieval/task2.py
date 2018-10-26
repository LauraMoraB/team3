import numpy as np 

##METHOD x2Distance
def x2Distance(histTarget, histSource):
    diference = 0.0
    for i in range(0,len(histSource[:,0])-1):
        uperDiv = (histSource[i,0]-histTarget[i,0])**2  + 1
        lowerDiv = histSource[i,0]+histTarget[i,0] + 1

        division = np.divide(uperDiv,lowerDiv)
        diference = diference + division
    return diference

##METHOD Histogram Intersection
def histIntersection(histTarget, histSource):
    intesection =0.0
    for i in range(0,len(histSource[:,0])-1):
        intesection = intesection + min(histSource[i,0],histTarget[i,0])
    return intesection

##METHOD Hellinger Kernel
def hellingerKernel(histTarget, histSource):
    similarity =0.0
    for i in range(0,len(histSource[:,0])-1):
        similarity = similarity + np.sqrt(histSource[i,0]*histTarget[i,0])
    return similarity

