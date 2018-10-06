
# coding: utf-8

# In[197]:


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

def create_df(path_ds_images, path_ds_masks, path_ds_gt):
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
        listGT.append(load_annotations(path_ds_gt+"/"+annot))
        
        
    col = ['UpLeft(Y)','UpLeft(X)','DownRight(Y)','DownRight(X)','Type', "Image", "Mask"]
    df = pd.DataFrame(columns=col)
    list=[]
    image = 0
    gt= 0
    dins =0
    for i in listGT:
        for j in range(len(listGT[image])):
            list = listGT[image][dins]
            list.append(listImages[image])
            list.append(listMask[image])
            df.loc[gt] = list
            dins = dins + 1
            gt = gt + 1
        dins = 0
        del list[:]
        list = []
        image = image + 1

    return df

