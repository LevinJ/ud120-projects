from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#data preparation
digits = datasets.load_digits()
features_train, features_test, labels_train, labels_test = train_test_split(digits.data, digits.target, random_state=42)
unique, counts = np.unique(labels_train, return_counts=True)
print np.asarray((unique, counts)).T


print "training number: %d, testing number: %d original feature num %d" %(len(labels_train), len(labels_test), len(features_train[0]))

clf = svm.SVC(kernel='rbf', C=1000, gamma=0.001)


clf.fit(features_train, labels_train)



y_pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,y_pred )
print "testing set accuracy is %f" %(accuracy)

#for training set
y_pred=clf.predict(features_train)
accuracy = accuracy_score(labels_train,y_pred )
print "training set accuracy is %f" %(accuracy)
