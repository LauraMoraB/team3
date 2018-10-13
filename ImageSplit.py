from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2


def compute_stats(df):
    #Stadistical study for the different signal types in order to properly
    #split the training set into two sets,  ~70% and ~30% with the best 
    #main features represented in both of them
    cols = ['Type', 'FillRatioMean', 'FillRatioStd', 'FormFactorMean', 'FormFactorStd', 'AreaMean', 'AreaStd', 'XMax', 'XMin', 'XMean', 'XStd', 'YMax', 'YMin', 'YMean', 'YStd']
    dfStats = pd.DataFrame(columns=cols)

    types = df.Type.unique().tolist()
    types.sort()
    fillRatioMean = []
    fillRatioStd = []
    formFactorMean = []
    formFactorStd = []
    areaMean = []
    areaStd = []    
    xMean = []
    xStd = []      
    xMax = []
    xMin = []    
    yMean = []
    yStd = []
    yMax = []
    yMin = []        
    
    for typeSignal in types:
        typeDf = df[df.Type == typeSignal]
        fillRatioMean.append(np.mean(typeDf['FillRatio']))
        fillRatioStd.append(np.std(typeDf['FillRatio']))
        formFactorMean.append(np.mean(typeDf['FormFactor']))
        formFactorStd.append(np.std(typeDf['FormFactor']))
        areaMean.append(np.mean(typeDf['Area']))
        areaStd.append(np.std(typeDf['Area']))
        xMean.append(np.mean(typeDf['X']))
        xStd.append(np.std(typeDf['X']))        
        xMax.append(np.max(typeDf['X']))
        xMin.append(np.min(typeDf['X'])) 
        yMean.append(np.mean(typeDf['Y']))
        yStd.append(np.std(typeDf['Y']))        
        yMax.append(np.max(typeDf['Y']))
        yMin.append(np.min(typeDf['Y'])) 
        
    dfStats['Type'] = types
    dfStats['FillRatioMean'] = fillRatioMean
    dfStats['FillRatioStd'] = fillRatioStd
    dfStats['FormFactorMean'] = formFactorMean
    dfStats['FormFactorStd'] = formFactorStd
    dfStats['AreaMean'] = areaMean
    dfStats['AreaStd'] = areaStd
    
    dfStats['XMax'] = xMax
    dfStats['XMin'] = xMin
    dfStats['XMean'] = xMean
    dfStats['XStd'] = xStd
    
    dfStats['YMax'] = yMax
    dfStats['YMin'] = yMin
    dfStats['YMean'] = yMean
    dfStats['YStd'] = yStd
        
    return dfStats


def plot_stats(df):        
    types = df.Type.unique().tolist()
    types.sort()
    plotting = ['FillRatio', 'FormFactor', 'Area']
    colors = ['k', 'r', 'g', 'b', 'm', 'c', 'y']
    for typeSignal in types:
        dfType = df[df.Type == typeSignal]
        for chart in plotting:
            data = dfType[chart].tolist()      
            plt.hist(data, bins=30, color = colors[0])
            plt.ylabel('f')
            plt.xlabel(chart)
            plt.title('signalType '+typeSignal)
            plt.show()
        colors.pop(0)
    
def sort_by_mean(reference, data):
    dataError = []
    meanData = np.mean(data)

    for value in data:
        dataError.append(abs(value - meanData))
    sortedReference = [x for _,x in sorted(zip(dataError, reference))]

    return sortedReference
        

def split_by_type(df, pathimages, pathmask):
    # Prepares train and valid dataframes to follow df structure
    col = list(df)
    train = pd.DataFrame(columns=col)
    validation = pd.DataFrame(columns=col)
    # Divides df images for signal type
    for typeSignal in df.Type.unique():
        typeDf = df[df.Type == typeSignal]
        # Sorts out subDf according to their signal area size
        reference = sort_by_mean(typeDf.index.values.tolist(), typeDf.Area.tolist())
        # Divides 30 - 70
        k = 0
        for indexRef in reference:
            if(k == 2 or k == 5 or k == 8):
                validation = validation.append(typeDf[typeDf.index == indexRef])
            else:
                train = train.append(typeDf[typeDf.index == indexRef])
            if(k == 9):
                k = 0
            else:
                k += 1                
    # Saves test and validation images in new subfolders
    for image in validation["Image"].tolist():  
        imageTrain = cv2.imread(pathimages+image,1)
        cv2.imwrite("./datasets/split/validation/"+image, imageTrain)
        
    for mask in validation["Mask"].tolist():
        maskTrain = cv2.imread(pathmask+mask,1)
        cv2.imwrite("./datasets/split/validation/mask/"+mask, maskTrain)
        
    for image in train["Image"].tolist():  
        imageTrain = cv2.imread(pathimages+image,1)
        cv2.imwrite("./datasets/split/train/"+image, imageTrain)
        
    for mask in train["Mask"].tolist():
        maskTrain = cv2.imread(pathmask+mask,1)
        cv2.imwrite("./datasets/split/train/mask/"+mask, maskTrain)
               
    return train, validation

    
        
        
        
        
    