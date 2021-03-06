#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "training number: %d, testing number: %d original feature num %d" %(len(labels_train), len(labels_test), len(features_train[0]))


#########################################################
### your code goes here ###

#########################################################
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clf = SVC(kernel='rbf',C=10000.0)
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
y_pred=clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
# print "10:%d,26:%d,50:%d" %(y_pred[10],y_pred[26],y_pred[50])
# print "sum of chris", y_pred.sum()

accuracy = accuracy_score(labels_test,y_pred )
print "testing set accuracy is %f" %(accuracy)

#for training set
y_pred=clf.predict(features_train)
accuracy = accuracy_score(labels_train,y_pred )
print "training set accuracy is %f" %(accuracy)

