# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 20:13:44 2016

@author: jianz
"""
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    total = len(labels_test)
    correct = (pred == labels_test).sum()
    accuracy = correct/float(total)
    from sklearn.metrics import accuracy_score
    
    accuracy = accuracy_score(labels_test,pred )
    return accuracy