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
    clf = SVC(kernel='rbf', C=100000, gamma=100)
    clf.fit(features_train, labels_train)
    return clf
        
    
