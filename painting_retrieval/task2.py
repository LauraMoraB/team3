import numpy as np 

##METHOD x2Distance
def x2Distance(histTarget, histSource):
    diference = 0.0
    for i in range(len(histSource)):
        uperDiv = (histSource[i]-histTarget[i])**2  + 1
        lowerDiv = histSource[i]+histTarget[i] + 1

        division = np.divide(uperDiv,lowerDiv)
        diference = diference + division
    return diference

##METHOD Histogram Intersection
def histIntersection(histTarget, histSource):
    intesection =0.0
    for i in range(len(histSource)):
        intesection = intesection + min(histSource[i],histTarget[i])
    return intesection

##METHOD Hellinger Kernel
def hellingerKernel(histTarget, histSource):
    similarity =0.0
    for i in range(len(histSource)):
        similarity = similarity + np.sqrt(histSource[i]*histTarget[i])
    return similarity

