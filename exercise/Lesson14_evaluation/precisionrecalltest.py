import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
labels_test = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])

print "true positive: %f" % (((y_pred == 1) & (labels_test==1)).sum())

print "precision socre: %f" % (precision_score(labels_test, y_pred, average='binary'))   
print "recall socre: %f" % (recall_score(labels_test, y_pred, average='binary')) 
