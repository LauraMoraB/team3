"""
Method for computing FLANN
"""
import cv2

def matcher(descA, descB, method):
    
    if method ==  "ORB":
        FLANN_INDEX_LSH = 6
        
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    else:   
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    
    # Iteracions
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    return flann.knnMatch(descA, descB, 2)
    

def Flann(top, siftA, siftB, method, th=0.5):
    imNameA, kpsA, descsA = siftA    
    imNameB, kpsB, descsB = siftB   

    distance = []
    
    if method == "DAISY":
        matches =  matcher(descsA[1], descsB[1], method)
    else:
        matches =  matcher(descsA, descsB, method)
    
    for i, m_n in enumerate(matches):
        if len(m_n) ==2:
            (m,n)=m_n
            if m.distance <= th*n.distance:
                distance.append(m.distance)
                    
    #distance = [ m.distance  for i,(m,n) in enumerate(matches) if m.distance <= th*n.distance]    
    
    distance.sort(reverse=True)
   
    distance = distance[0:top]
    
    return distance

def retreive_image_withFlann(siftDs, siftQueries, paths, k, method, th = 60, descsMin = 3):  
    
    queriesResult = []
    distancesResult = []
    
    finalMatch=[]
    
    for imNameQuery in siftQueries:
        matches = []

        siftQuery = siftQueries[imNameQuery]
     
        for imNameDs in siftDs:
            
            siftIm = siftDs[imNameDs]
            
            distance =  Flann(100, siftQuery, siftIm, method, th)
            
            matches.append([imNameDs, distance])

        # Sort images per number of matches under threshold level
        matches = sorted(matches, key = lambda x:len(x[1]), reverse = True)
        
        if(len(matches) > k):
            matches = matches[0:k]
        
        # Detect if image is not present in the Dataset
        tots=0
        for index,row in enumerate(matches):
            # Comprobar si tots son mes petits a un threshold
            if len(row[1]) < descsMin:
               tots+=1
              
        # Contruct query result to be returend
        distancesResult.append([ row[1] for row in matches ])
        queriesResult.append([ row[0]  for row in matches ] if tots<10 else [-1])
        
        finalMatch.append(matches)
    
    
    return queriesResult, distancesResult, finalMatch
