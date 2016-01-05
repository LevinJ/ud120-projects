# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 13:25:31 2016

@author: jianz
"""

#Gaussian Naive Bayes

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
gnb = GaussianNB()
gnbmodel =  gnb.fit(iris.data, iris.target)
y_pred = gnbmodel.predict(iris.data)
total = iris.data.shape[0]
mislabled = (iris.target != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d;error rate:%f" % (total,mislabled, mislabled/float(total)))


digits = datasets.load_digits()
y_pred = gnb.fit(digits.data, digits.target).predict(digits.data)
total = digits.data.shape[0]
mislabled = (digits.target != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d;error rate:%f" % (total,mislabled, mislabled/float(total)))