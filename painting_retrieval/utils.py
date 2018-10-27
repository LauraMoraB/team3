import pandas as pd
import pickle
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

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

def create_dir(pathSave):
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)

def get_full_image(dfSingle, path):
    return cv2.imread(path + dfSingle['Image'],1)
	
	
def get_image(im, path):
    return cv2.imread(path+im, 1)

def plot_gray(im):
    plt.imshow(im, cmap='gray')
    plt.show()

def plot_rgb(im):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()

def save_pkl(list_of_list, path):
    create_dir(path)
    with open(path+'results.pkl', 'wb') as f:
        pickle.dump(list_of_list, f)

def submission_list(df):
    project_result = []
    for query in df.Query.unique():
        dfQuery = df[df.Query == query]
        query_result = []
        for i in range(len(dfQuery)):
            dfImg = dfQuery[dfQuery.Order == i]
            query_result.append(dfImg['Image'].values[0])
        project_result.append(query_result.copy())
    return project_result

## Average precision at K calculation
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

## Mean average precision at K calculation
def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])