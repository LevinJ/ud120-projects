import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()


clf = svm.SVC(kernel='linear', C=1)
# n_samples = iris.data.shape[0]
# cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.1, random_state=0)

# scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_validation.cross_val_predict(clf, iris.data,iris.target, cv=10)
print accuracy_score(iris.target, predicted) 