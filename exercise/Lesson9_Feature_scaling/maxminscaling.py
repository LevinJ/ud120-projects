""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
import numpy as np
from sklearn import preprocessing
def featureScaling1(arr):
    nparr = np.array(arr)
    nparr = nparr.astype(np.float64)
    max_value = nparr.max()
    min_value = nparr.min()
    if max_value == min_value:
        nparr.fill(0.5)
        return nparr
    
    nparr = (nparr - min_value )/(max_value -min_value) 
    return nparr

def featureScaling2(arr):
    nparr = np.array(arr)
    nparr = nparr.astype(np.float64)
    nparr = nparr.reshape((nparr.shape[0],1))
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(nparr)
    return X_train_minmax

def featureScalinge(arr):
    nparr = np.array(arr)
    nparr = nparr.astype(np.float64)
    nparr = nparr.reshape((nparr.shape[0],1))
    X_scaled = preprocessing.scale(nparr)
    return X_scaled
# tests of your feature scaler--line below is input data
# data = [115, 115, 115]
data = [1., -1.,  2.]
print featureScaling1(data)
print featureScaling2(data)
print featureScalinge(data)

