#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
import numpy as np
import pandas as pd


class Data_Exploration_Proxy:
    def testInit(self):
        with open("final_project_dataset.pkl", "r") as data_file:
            data_dict = pickle.load(data_file)
        features_list = data_dict['ALLEN PHILLIP K'].keys()
        features_list.remove('poi')
        features_list.insert(0, 'poi')
        my_dataset = data_dict
        data = featureFormat(my_dataset, features_list, remove_NaN=False, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        return labels, features,features_list
    def getBasicStatistics(self):
        labels,features,features_list = self.testInit()
        self.savetoCSV(labels,features,features_list)
        features_df = pd.DataFrame(features, columns=features_list[1:])
        print features_df.describe()
        print "total number of data points: ", len(labels)
        print "allocation across classes (POI/non-POI)", np.array(labels).sum()/len(labels)
        print "number of total existing features", len(features_list)
        return
    def savetoCSV(self,labels,features,features_list):
        labels = np.array(labels).reshape((len(labels),1))
        data = np.hstack((labels, np.array(features)))
        df = pd.DataFrame(data, columns=features_list)
        df.to_csv('enron_data.csv')
        return
    def removeOutlier(self):
        return
    

# print features

def main():
    test =  Data_Exploration_Proxy() 
    test.getBasicStatistics()

if __name__ == '__main__':
    main()