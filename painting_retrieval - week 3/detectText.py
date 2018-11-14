import numpy as np
from matplotlib import pyplot as plt
import cv2
from resizeImage import image_resize
from utils import list_ds, create_dir, save_pkl
from joinSquare import compare_squares
from textValidation import validation_window


def plot_images(after_morph, border_del, final_total, image_with_square, final_image_color):
   
    plt.subplot(121), plt.imshow(after_morph)
    plt.title('After morphology Bin'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(122), plt.imshow(border_del, cmap='gray')
    plt.title('Borders removed'), plt.xticks([]), plt.yticks([])   
    plt.show()
    plt.imshow(final_total, cmap='gray')
    plt.title('Dilate to join'), plt.xticks([]), plt.yticks([])
    plt.show()
    

    plt.subplot(121), plt.imshow(cv2.cvtColor(image_with_square, cv2.COLOR_BGR2RGB))
    plt.title('Image Multiple Detection'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(122), plt.imshow(cv2.cvtColor(final_image_color, cv2.COLOR_BGR2RGB))
    plt.title('Image Final Detection'), plt.xticks([]), plt.yticks([])
    plt.show()   
    
def apply_morphology(img_back, perH, perW):
 
    # Define kernels
    kernelOpen = np.ones((10,10), np.uint8)
    kernelDilate  = np.ones((20,20),np.uint8)
    kernetEro  = np.ones((20,20),np.uint8)
    
    # Compute filters
    opened = cv2.morphologyEx(img_back, cv2.MORPH_OPEN, kernelOpen)

    dilated = cv2.dilate(opened,kernelDilate, iterations = 1)

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
    h_mid = h//2
    w_mid = w//2
    
    ## compute FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
   # magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    # Filter to remove low freq
    fshift[h_mid-50:h_mid+50, w_mid-50:w_mid+50] = 0
    
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

def final_detection(image, lista):
    x1=lista[0]
    y1=lista[1]
    x2=lista[2]
    y2=lista[3]
    
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),10)

def draw_square(binary_image, color_image, w_image, h_image, perH, perW):
   
    contours = find_contours(binary_image)
    detection=[]
    
    if contours:
        for index, cnt in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            if area > 0.01*(w_image*h_image):
                if w>2*h:
                    
                    x1=int(x*perW)
                    x2=int((x+w)*perW)
                    y1=int(y*perH)
                    y2=int((y+h)*perH)

                    
                    detection.append([x1, y1, x2, y2])
                    cv2.rectangle(color_image,(x1,y1),(x2,y2),(0,255,0),10)
                
    
    return color_image, detection

def validate_results(pathGT, pathResults):
    validation_window(pathGT, pathResults)


def detect_text_bbox(pathDS, plot):
    
    list_of_detections_total = []
    im_list = list_ds(pathDS)
    
    #im_list = ["ima_000002.jpg"]
    for imName in im_list:
        print (imName)
        img = cv2.imread(pathDS+imName)
        final_image = img.copy()
                
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        (h_image_complete, w_image_complete) = gray.shape
        
        # resize
       # gray = image_resize(gray, 500)
#        res = image_resize(final_image, 500)

        (h_image, w_image) = gray.shape
       
        percentatge_reduccio_h =  h_image_complete/h_image
        percentatge_reduccio_w = w_image_complete/w_image
        
        img_back = high_pass(gray, h_image, w_image)
        
        plt.imshow(img_back)
        plt.show()
        
        # Morph
        after_morph = apply_morphology(img_back, percentatge_reduccio_h, percentatge_reduccio_w)
        
        #Return contours filled
        thresh = contour_detection(after_morph)
        
        # Delete components in the surroundings   
        delete_borders(thresh, w_image, h_image, w_per=0.1, h_per=0.01)
        
        kernelDil  = np.ones((1,60),np.uint8)
        
        # Final Image
        final_total_bin = cv2.dilate(thresh, kernelDil, iterations=1)
        
        
        image_with_square, list_of_detections = draw_square(final_total_bin, img, w_image, h_image, percentatge_reduccio_h, percentatge_reduccio_w)
        
        
        if len(list_of_detections) > 0:
            
            # Join boxxes if they are really close
            index = 0
                  
            while (index < len(list_of_detections)-1):
                current_value = list_of_detections[index]
                next_value = list_of_detections[index+1]
                
                # compare distances for width
                rectangle, joined = compare_squares(current_value, next_value)
                
                if joined:
                    list_of_detections.pop(index+1)
                    list_of_detections[index]=rectangle
                else:
                    index+=1
  
            # Get the bigger one and remove others
            area = np.array([ (item[2]-item[0])*(item[3]-item[1]) for item in list_of_detections])  
            ids = np.argsort(-area)
            list_of_detections = list_of_detections[ids[0]]
            
            
            # Draw final Square
            print ("After: ", list_of_detections)
            final_detection(final_image, list_of_detections)
       
        
        list_of_detections_total.append(list_of_detections)
    
        if plot == True:
            plot_images(after_morph, thresh, final_total_bin, image_with_square, final_image)
            
            
    return list_of_detections_total  
    
if __name__ == "__main__":   
    
    list_of_text_bbox = detect_text_bbox("dataset/", plot=True)
    
    # save pkl
    #save_pkl(list_of_text_bbox, "TextResults/")
       
        
        
