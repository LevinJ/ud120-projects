#!/usr/bin/python

import pickle
import numpy as np
from time import time
from sklearn import tree
np.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
print "training sample number: ", len(features_train)
clf = tree.DecisionTreeClassifier()

print "start training..."
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#for testing set
t0 = time()
accuracy=clf.score(features_test,labels_test)
print "prediction time:", round(time()-t0, 3), "s"

print "testing set accuracy is %f" %(accuracy)
# 
# #for training set
# y_pred=clf.predict(features_train)
# accuracy = accuracy_score(labels_train,y_pred )
accuracy = clf.score(features_train,labels_train)
print "training set accuracy is %f" %(accuracy)

#list important features

importantfea = clf.feature_importances_ 
#iterate the loop
feanum=[]
for i, val in enumerate(importantfea):
    if val > 0.2:
        print i, val
        feanum.append(i)
        
# orderedfealist = importantfea[np.argsort(importantfea)]
# orderedfealist = np.flipud(orderedfealist)
# print orderedfealist
print np.array(vectorizer.get_feature_names())[feanum]





