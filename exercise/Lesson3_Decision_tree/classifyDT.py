# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 19:26:25 2016

@author: jianz
"""
from sklearn import tree

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    clf = tree.DecisionTreeClassifier(min_samples_split = 50)
    clf.fit(features_train, labels_train)
    return clf
        
    
