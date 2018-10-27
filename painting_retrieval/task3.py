from task2 import  hellingerKernel, histIntersection, x2Distance
import operator


def getX2results(histogram_list_dataset, histogram_query, K, dfDataset):
    
    dictList = {}
    counter = 0
    for histogram_dataset in histogram_list_dataset: 
        
        name = dfDataset['Image'].iloc[counter]
        counter+=1
        distance = x2Distance(histogram_dataset, histogram_query)
        dictList[name] = distance

    pairedList = sorted(dictList.items(), key = operator.itemgetter(1))
    distanceList = list(zip(*pairedList))[0]
    distanceList =  distanceList[:K]
    
    return list(distanceList)
        
def getHellingerKernelResult(histogram_list_dataset, histogram_query, K, dfDataset):
    dictList = {}
    counter = 0
    for histogram_dataset in histogram_list_dataset: 
        
        name = dfDataset['Image'].iloc[counter]
        counter+=1
        distance = hellingerKernel(histogram_dataset, histogram_query)
        dictList[name] = distance

    pairedList = sorted(dictList.items(), key = operator.itemgetter(1))
    simlarityList = list(zip(*pairedList))[0]
    simlarityList =  simlarityList[:K]
    return list(simlarityList)

def getHistInterseccionResult(histogram_list_dataset, histogram_query, K, dfDataset):
    dictList = {}
    counter = 0
    for histogram_dataset in histogram_list_dataset: 
        
        name = dfDataset['Image'].iloc[counter]
        counter+=1
        distance = histIntersection(histogram_dataset, histogram_query)
        dictList[name] = distance

    pairedList = sorted(dictList.items(), key = operator.itemgetter(1))
    intersectionList = list(zip(*pairedList))[0]
    intersectionList =  intersectionList[:K]
    return list(intersectionList)
