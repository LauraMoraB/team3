import cv2
from ImageFeature import get_cropped_masked_image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def BGR2HSV(image):
    # converts RGB image array into H, S, V, unidimensional ordered list
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
  
def compute_HS_mask(img, h_range, s_range):  
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    :
    return cv2.inRange(hsv, (h_range[0], h_range[1], 0), (h_range[0], h_range[1],255))

def get_px_values(dfSingle, path):
    # return hue and sat values for valid px  
    image = get_cropped_masked_image(dfSingle, path)
    hueL, satL, valL = BGR2HSV(image)
    imageL = image2list(image)
    validHue = []
    validSat = []
#    validBGR = []
    pxCount = 0
    for px in imageL:
        if any(px):
            validHue.append(hueL[pxCount])
            validSat.append(satL[pxCount])
#            validBGR.append(imageL[pxCount])
        pxCount += 1
#    return validBGR
    return (validHue, validSat)

def get_color_histogram(df, path, number_bins):
    # creates color histograms in HSV for each signalType in the df
    totalHueL = []
    totalSatL = []
    
    edges = np.linspace(0, 255, number_bins+1)
    edges_log = np.logspace(0, 1, number_bins+1)
    edges_log = (edges_log-1)*255/max(edges_log)
    bin_center =(edges[:-1]+edges[1:])/2
    bin_center_log =(edges_log[:-1]+edges_log[1:])/2
    
    for typeSignal in sorted(df.Type.unique()):
        typeDf = df[df.Type == typeSignal]    
        hueL = []
        satL = []
        for i in range(len(typeDf)):
            dfSingle = typeDf.iloc[i]     
            (hueSingle, satSingle) = get_px_values(dfSingle, path)
            hueL.extend(hueSingle)
            satL.extend(satSingle)

        totalHueL.extend(hueL) 
        totalSatL.extend(satL)   
   
        (hist, bin_edges_x, bin_edges_y) = np.histogram2d(hueL, satL, bins = [edges, edges_log])
        (thetaL, rL, area) = plot_polar_histogram2D(hist, bin_center, bin_center_log, typeSignal)

    (hist, bin_edges_x, bin_edges_y) = np.histogram2d(totalHueL, totalSatL, bins = [edges, edges_log])
    (thetaL, rL, area) = plot_polar_histogram2D(hist, bin_center, bin_center_log, 'All')
        
    
def plot_polar_histogram2D(histN, binCenterX, binCenterY, typeSignal):
    
    map_rad = np.vectorize(lambda x: (2*np.pi*x/255))    
    theta = map_rad(binCenterX)
    thetaL = np.append([],[theta for i in range(len(histN))]) 
    r = map_rad(binCenterY)
    rL = []
    for i in range(len(histN)):
        for j in range(len(histN)):
            rL = np.append(rL, r[i])

    histL = image2list(histN)
    area_factor = 100
    area = list(map(lambda x: area_factor*x/max(histL) ,histL))

    for i in reversed(range(len(histL))):
        if(area[i]<(10)): 
            thetaL=np.delete(thetaL, i)
            rL=np.delete(rL, i)
            area=np.delete(area, i)           

    norm = mpl.colors.Normalize(0.0, 2*np.pi)
    colormap = plt.get_cmap('hsv')
    ax = plt.subplot(111, polar=True)
    ax.scatter(thetaL, rL, c=thetaL, s=area, cmap=colormap, norm = norm, alpha=0.75, edgecolor = 'black')
    plt.yticks([])
    plt.title('Signal type: '+typeSignal)
    ax.set_rmax(2)
    ax.set_rorigin(-.3)
    plt.savefig('C:/GitHub/team3/histograms/'+typeSignal+'.png')
    plt.show()
    
    return thetaL, rL, area