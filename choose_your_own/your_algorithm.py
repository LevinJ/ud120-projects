#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from pre_algorithms import getClf
from sklearn.metrics import accuracy_score
from time import time
from sklearn.cross_validation import cross_val_score
from PIL import Image 
import numpy as np


features_train, labels_train, features_test, labels_test = makeTerrainData()

print "training number: %d, testing number: %d original feature num %d" %(len(labels_train), len(labels_test), len(features_train[0]))

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

clf = getClf()

print "start training..."
t0 = time()
clf = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#for testing set
t0 = time()
y_pred=clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
accuracy = accuracy_score(labels_test,y_pred )
print "testing set accuracy is %f" %(accuracy)

#for training set
# y_pred=clf.predict(features_train)
# accuracy = accuracy_score(labels_train,y_pred )
accuracy = clf.score(features_train,labels_train)
print "training set accuracy is %f" %(accuracy)


#analyze adaboost
# plt.figure(2)
# class_names = "AB"
# plot_colors = "br"
# twoclass_output = clf.decision_function(features_test)
# plot_range = (twoclass_output.min(), twoclass_output.max())
# for i, n, c in zip(range(2), class_names, plot_colors):
#     plt.hist(twoclass_output[np.array(labels_test) == i],
#              bins=10,
#              range=plot_range,
#              facecolor=c,
#              label='Class %s' % n,
#              alpha=.5)
# x1, x2, y1, y2 = plt.axis()
# plt.axis((x1, x2, y1, y2 * 1.2))
# plt.legend(loc='upper right')
# plt.ylabel('Samples')
# plt.xlabel('Score')
# plt.title('Decision Scores')
# plt.show()

try:
    prettyPicture(clf, features_test, labels_test)
    image = Image.open("test.png")
    image.show()
except NameError:
    pass
