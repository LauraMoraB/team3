import pandas as pd
import pickle
import os
import cv2
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


## Save PKL
# Revisar inputs
def save_pkl(list_of_list, path):
    with open(path+'results.pkl', 'wb') as f:
        pickle.dump(list_of_list, f)

def get_image(df, path):
    return cv2.imread(path + df['Image'], 1)

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

         

