import pandas as pd
import pickle
import os
import cv2
from matplotlib import pyplot as plt

def create_df(path_images):
    listImages =[]
    # Import Data from directories
    for filename in os.listdir(path_images):
        if filename.endswith(".jpg"):
            listImages.append(filename)
    col = ["Image"]
    df = pd.DataFrame(columns=col)
    df["Image"] = listImages
    return df

## Save PKL
# Revisar inputs
def create_dir(pathSave):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    
def save_pkl(list_of_list,path):
    create_dir(path)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(list_of_list, f)

def get_image(im, path):
    return cv2.imread(path+im, 1)

def plot_gray(im):
    plt.imshow(im, cmap='gray')
    plt.show()
    
def plot_rgb(im):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()