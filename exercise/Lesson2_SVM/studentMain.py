# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 19:27:46 2016

@author: jianz
"""

#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from Classifyfunc import classify
from sklearn.metrics import accuracy_score



features_train, labels_train, features_test, labels_test = makeTerrainData()

print "training number: %d, testing number: %d original feature num %d" %(len(labels_train), len(labels_test), len(features_train[0]))

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

clf = classify(features_train, labels_train)

# print NBAccuracy(features_train, labels_train, features_test, labels_test)

#for testing set

y_pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,y_pred )
print "testing set accuracy is %f" %(accuracy)

#for training set
y_pred=clf.predict(features_train)
accuracy = accuracy_score(labels_train,y_pred )
print "training set accuracy is %f" %(accuracy)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())