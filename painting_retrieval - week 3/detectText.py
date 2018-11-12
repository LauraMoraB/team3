import numpy as np
from matplotlib import pyplot as plt
import cv2
from resizeImage import image_resize
from utils import list_ds, create_dir


def plot_images(thresh, final, image_with_square):
    plt.subplot(121), plt.imshow(thresh)
    plt.title('After morphology Bin'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(final, cmap='gray')
    plt.title('Image Final'), plt.xticks([]), plt.yticks([])   
    plt.show()
    

    plt.imshow(cv2.cvtColor(image_with_square, cv2.COLOR_BGR2RGB))
    plt.title('Image With Detection'), plt.xticks([]), plt.yticks([]) 
    plt.show()   
    
def apply_morphology(img_back):
    
    kernelOpen = np.ones((10,10), np.uint8) 
    opened = cv2.morphologyEx(img_back, cv2.MORPH_OPEN, kernelOpen)    
    
    kernelDilate  = np.ones((20,20),np.uint8)
    dilated = cv2.dilate(opened,kernelDilate, iterations = 1)
    
    kernetEro  = np.ones((20,20),np.uint8)
    erode = cv2.erode(dilated, kernetEro, iterations=1)
    
    
    final = cv2.dilate(erode, kernelDilate, iterations=1)
    return final
    
    
def delete_borders(image, w_image, h_image, w_per=0.1, h_per=0.05):
    
    limit_w = int(w_image*w_per)
    limit_h = int(h_image*h_per)
    
    for col in range(0, limit_w):
        image[:,col]=0
        image[:,w_image-1-col]=0
    
    for row in range(0, limit_h):
        image[row,:]=0
        image[h_image-row-1,:]=0    

    
def high_pass(img, h, w):    
    h_mid = h // 2
    w_mid = w//2
    
    ## compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
   # magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    # Filter to remove low freq
    fshift[h_mid-30:h_mid+30, w_mid-30:w_mid+30] = 0
    
    # Compute Inverse FFT ang get image
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back.astype(np.uint8)


def find_contours(imgray):
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)
    heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_detection(image_processed):
    
    blur = cv2.blur(image_processed, (5, 5), 0)    
    contours = find_contours(blur)
    if contours:
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.drawContours(image_processed, [cnt], -1, (255,255,255), thickness=cv2.FILLED)  
            
    # Fem un filled dels contorns i ho passema  binary xk sigui una mascara
    ret, thresh = cv2.threshold(image_processed, 254,255,cv2.THRESH_BINARY)
    
    return thresh


def draw_square(binary_image, color_image, w_image, h_image):
    
    contours = find_contours(binary_image)
    detection=[]
    if contours:
        for index, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            
            # check aspect ratio and area
 
            # compute area
            area = w*h
            if area > 0.01*(w_image*h_image):
                if w>2*h:
                    print ("Area bigger!")
                    detection = [x,y,x+w,y+h]
                    cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),10)
                
    
    return color_image, detection

if __name__ == "__main__":   
    
    im_list = list_ds("dataset/")
    plot = False
    list_of_detections_total = []
    
    for imName in im_list:
       
        img = cv2.imread("dataset/"+imName)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h_image, w_image) = gray.shape
        
        img_back = high_pass(gray, h_image, w_image)
        
        # Morph
        final = apply_morphology(img_back)
        
        #Return contours filled
        thresh = contour_detection(final)
        
        # Delete components in the surroundings   
        delete_borders(final, w_image, h_image, w_per=0.1, h_per=0.05)
        
        kernelDil  = np.ones((10,60),np.uint8)
        final = cv2.dilate(final, kernelDil, iterations=1)
        
        image_with_square, list_of_detections = draw_square(final, img, w_image, h_image)
        list_of_detections_total.append(list_of_detections)
        
        if plot == True:
            plot_images(thresh, final, image_with_square)
       
