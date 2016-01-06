# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 19:26:25 2016

@author: jianz
"""
from sklearn.svm import SVC

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
#     clf = SVC(kernel='rbf', C=1, gamma=10)
    clf = SVC(kernel='rbf',C=10,gamma=200)
    clf.fit(features_train, labels_train)
    return clf
        
    
# kernel='rbf'
# 
# testing set accuracy is 0.920000
# training set accuracy is 0.924000
# 
# 
# kernel='rbf',gamma=10
# testing set accuracy is 0.932000
# training set accuracy is 0.95866
# 
# kernel='rbf',gamma=100
# testing set accuracy is 0.936000
# training set accuracy is 0.966667
# 
# 
# kernel='rbf',C=10,gamma=1000
# testing set accuracy is 0.916000
# training set accuracy is 0.996000
# 
# kernel='rbf',C=10,gamma=500
# testing set accuracy is 0.932000
# training set accuracy is 0.982667
# 
# kernel='rbf',C=10,gamma=300
# testing set accuracy is 0.932000
# training set accuracy is 0.982667
# 
# kernel='rbf',C=10,gamma=200
# testing set accuracy is 0.948000
# training set accuracy is 0.977333
# 
# kernel='rbf',C=8,gamma=200
# testing set accuracy is 0.948000
# training set accuracy is 0.977333