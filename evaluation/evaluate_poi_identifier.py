#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)

labels_test = np.array(labels_test)
print "test set: %f poi out of %f total test samples" % (labels_test.sum(), labels_test.shape[0] )
print "true positive: %f" % (((y_pred == 1) & (labels_test==1)).sum())

print "precision socre: %f" % (precision_score(labels_test, y_pred, average='binary'))   
print "recall socre: %f" % (recall_score(labels_test, y_pred, average='binary')) 


