#!/usr/bin/python

import numpy as np
# y = np.array([[11, 12, 13], [51, 52, 53]])
# x = y.T
# a = np.array([1, 2, 3])
# b = np.array([5, 6,7])
# c = np.array([8,9,10])
# # d =np.hstack((a,b,c,x))
# d =np.column_stack((a,b,c,x))
# print d
# d1= d.tolist()
# print d1
# 
# x1 = (a - b) ** 2
# print x1;
# a = np.array([[1,4], [3,1]])
# a.sort(axis=1)
# a.sort(axis=0)

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    remained_len = len(net_worths) - 0.1 * len(net_worths)
    errors = (predictions - net_worths) ** 2
    remained_indices = errors[:,0].argsort()[0:remained_len]
    cleaned_data = np.column_stack((ages[remained_indices],net_worths[remained_indices],errors[remained_indices])).tolist()
#     print errors

    ### your code goes here

    
    return cleaned_data

