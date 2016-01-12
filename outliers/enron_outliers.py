#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL', 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
def isNaN(num):
    return num == "NaN"
def findLargestOutlier():
    for key, value in data_dict.iteritems():
        if not (isNaN(value['salary']) and isNaN(value['bonus'])):
            if (value['salary']> 1e6 and value['bonus']> 5e6):
                print "maxkey", key, value['salary'], value['bonus']
    return

findLargestOutlier()
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


