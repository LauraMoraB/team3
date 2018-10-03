import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import csv


def load_annotations(annot_file):
    # Annotations are stored in text files containing
    # the coordinates of the corners (top-left and bottom-right) of
    # the bounding box plus an alfanumeric code indicating the signal type:
    # tly, tlx, bry,brx, code
    annotations = []
    signs       = [] 

    for line in open(annot_file).read().splitlines():

        annot_values = line.split()
        annot_values = [x.strip() for x in annot_values]
        for ii in range(4):
            annot_values[ii] = float(annot_values[ii])
        annotations.append(annot_values)
        
    return annotations



def create_df(path_images, path_masks, path_gt):
    path_ds_images=path_images
    path_ds_masks=path_masks
    path_ds_gt=path_gt
    
    listImages =[]
    listMask=[]
    listGT=[]

    # Import Data from directories
    for filename in os.listdir(path_ds_images):
        if filename.endswith(".jpg"):
            listImages.append(filename)

    for mask in os.listdir(path_ds_masks):
        listMask.append(mask)

    for annot in os.listdir(path_ds_gt):
        listGT.append(load_annotations(path_ds_gt+"/"+annot)[0]) #se pone 0 al final para quitar la dimension de mas
        
    # Create DataFrame from lists
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type']
    df = pd.DataFrame(listGT,columns=col)
    df['Imagen']=listImages
    df['Mask']=listMask
    
    return df

