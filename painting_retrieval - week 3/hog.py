from utils import list_ds, get_gray_image, plot_gray
import cv2
import numpy as np


    
def compute_HOG(image):
    edges = auto_canny(image)  
    plot_gray(edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,100,None, 0, 0)
    if lines is not None:
       for line in lines[0:9]:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image
            
def list_images(path, resize = False):
    hog_images = []
    im_list = list_ds(path)

    for imName in im_list:
        
        image = get_gray_image(imName, path, resize)
        hog_image= compute_HOG(image)
        plot_gray(image)
        hog_images.append(hog_image)
    
    return hog_images

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged