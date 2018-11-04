import cv2
import matplotlib.pyplot as plt

def feature_detection(method, im):
    minHessian = 400
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
    kp =  detector.detect(im)
    
    return kp

def compute_daisy(im):
    # Detecte Features
    keypoints = feature_detection("SURF", im)
    daisy = cv2.xfeatures2d.DAISY_create()
    
    desc = daisy.compute(im, keypoints)

    return desc, keypoints 

def matcher(descA, descB):
    
   # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descA,descB,2)
    return matches
    
    
    
def init():
    imA = cv2.imread("dataset/ima_000115.jpg",0)
    imB = cv2.imread("queries_validation/ima_000002.jpg",0)
    
    descA,kpA = compute_daisy(imA)
    descB,kpB = compute_daisy(imB)
    
    matches =  matcher(descA[1], descB[1])
    
    matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)
    
    img3 = cv2.drawMatchesKnn(imA,kpA,imB,kpB,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
    
    return matches