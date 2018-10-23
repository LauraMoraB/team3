import pandas as pd
import pickle
import os

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
def save_pkl(list_of_list,path):
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(list_of_list, f)

def get_image(df, path):
    return cv2.imread(path + df['Image'], 1)