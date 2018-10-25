from task1 import get_image, compute_histogram

def store_histogram_total(df, channel1, channel2, channel3):
    """
    Method to save histograms if necessary
    """
    df[channel1] = ""
    df[channel2] = ""
    df[channel3] = ""

    channels = [0,1,2]
    hist_one=[]
    hist_total=[]
    
    for i in range(len(df)):
        im = get_image(df.iloc[0], "dataset/")
        
        for c in channels:
            hist =  compute_histogram(im, c)
            hist_one.append(hist)
            
        hist_total.append(hist_one)
             
        df[channel1].iloc[i] = hist_total[i][0]
        df[channel2].iloc[i] = hist_total[i][1]
        df[channel3].iloc[i] = hist_total[i][2]    
        
    return df