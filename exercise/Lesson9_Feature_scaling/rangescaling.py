from sklearn import preprocessing
import numpy as np
### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    arr = np.array(arr).reshape(len(arr),1).astype(np.float64)
    min_max_scaler = preprocessing.MinMaxScaler()
    arr =  min_max_scaler.fit_transform(arr)
    arr = arr.flatten()
    return arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)
