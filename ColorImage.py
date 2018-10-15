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
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([h_range[0], s_range[0], 0]), np.array([h_range[1], s_range[1], 255]))

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
    redHueL = []
    blueHueL = []
    totalSatL = []
    redSatL = []
    blueSatL = []
    
    # Lineas edges for Hue
    edges = np.linspace(0, 179, number_bins+1)
    # Log edges for Saturation since interested space is compreseed
    edges_log = np.logspace(0, 1, number_bins+1)
    edges_log = (edges_log-1)*255/max(edges_log)
    # Center for each bin is calculated
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
        if(typeSignal == 'D' or typeSignal == 'F'):
            blueHueL.extend(hueL) 
            blueSatL.extend(satL)  
        else:
            redHueL.extend(hueL) 
            redSatL.extend(satL)              
        # Plots histogram for each signal Type
        (hist, bin_edges_x, bin_edges_y) = np.histogram2d(hueL, satL, bins = [edges, edges_log])
        (histL, edgesXL, edgesYL) = process_histogram(hist, bin_center, bin_center_log)
        plot_polar_histogram(histL, edgesXL, edgesYL, typeSignal)
    # Plots histogram for red signals
    (hist, bin_edges_x, bin_edges_y) = np.histogram2d(redHueL, redSatL, bins = [edges, edges_log])
    (histL, edgesXL, edgesYL) = process_histogram(hist, bin_center, bin_center_log)
    plot_polar_histogram(histL, edgesXL, edgesYL, 'red')
    # Plots histogram for blue signals    
    (hist, bin_edges_x, bin_edges_y) = np.histogram2d(blueHueL, blueSatL, bins = [edges, edges_log])
    (histL, edgesXL, edgesYL) = process_histogram(hist, bin_center, bin_center_log)
    plot_polar_histogram(histL, edgesXL, edgesYL, 'blue')
    # Plots histogram for all signals        
    (hist, bin_edges_x, bin_edges_y) = np.histogram2d(totalHueL, totalSatL, bins = [edges, edges_log])
    (histL, edgesXL, edgesYL) = process_histogram(hist, bin_center, bin_center_log)
    plot_polar_histogram(histL, edgesXL, edgesYL, 'all')
    
def process_histogram(hist, edgesX, edgesY):
    
    dim = len(hist)
    # Serialize arrays
    histL = image2list(hist)
    edgesXL = np.append([],[edgesX for i in range(dim)]) 
    edgesYL = []
    for i in range(dim):
        for j in range(dim):
            edgesYL = np.append(edgesYL, edgesY[i])
    # Normalize histogram
    eHistLN = np.sum(histL)
    histLN = list(map(lambda x: x/eHistLN, histL))    
    # Remove outliers
    for i in reversed(range(len(histLN))):
        if(histLN[i]<(0.01)): 
            edgesXL=np.delete(edgesXL, i)
            edgesYL=np.delete(edgesYL, i)
            histLN=np.delete(histLN, i)         
    return (histLN, edgesXL, edgesYL)
    
def plot_polar_histogram(histN, binCenterX, binCenterY, typeSignal):
    
    map_hue = np.vectorize(lambda x: (2*np.pi*x/179))    
    map_sat = np.vectorize(lambda x: (2*np.pi*x/255))    
    thetaL = map_sat(binCenterX)
    rL = map_hue(binCenterY)
 
    areaFactor = 2
    area = areaFactor*((histN*100)**2)    

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
    