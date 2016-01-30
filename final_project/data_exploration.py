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
    def __init__(self,data_dict):
        self.data_dict = data_dict
        return
    def run(self):
#         features_list = self.__getAllFeatures()
        features_list = ['poi','salary','bonus']
        self.__dispBasicStatistics(features_list)
        self.__saveToCsv(features_list)
        return
    def __identifyAllZeros(self, features_list):
        #identify the record which has all zeros features
        numAllZeros = 0
        if features_list[0] == 'poi':
            features_list = features_list[1:]
        else:
            features_list = features_list
            
        for key in sorted(self.data_dict.keys()):
            isAllzeros = True
            for feature in features_list:
                if self.data_dict[key][feature] !=  'NaN':
                    isAllzeros = False
            if isAllzeros:
                print 'record ', key, ' is all zeroes and has been removed from samples'
                numAllZeros = numAllZeros + 1
        print "Total number of removed all zeros records; ", numAllZeros
        return
    def __getAllFeatures(self):
        features_list = self.data_dict['ALLEN PHILLIP K'].keys()
        features_list.remove('poi')
        features_list.insert(0, 'poi')
        features_list.remove('email_address')
        return features_list
    def __saveToCsv(self,features_list):
        print "Save selected data to csv"
        data = featureFormat(self.data_dict, features_list, sort_keys = True)
        df = pd.DataFrame(data, columns=features_list)
        df.to_csv('selecteddata.csv')
        return
    def __dispBasicStatistics(self, features_list):
        print "display basic statistics about the data set"
        data = featureFormat(self.data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        self.__identifyAllZeros(features_list)
        print "features used: ", features_list
        print "number of total existing features", features[0].shape[0]
        print "total number of data points: ", len(labels)
        print "total number of Poi", (np.array(labels) == 1).sum()
        print "total number of non_Poi", len(labels) - (np.array(labels) == 1).sum()
        print "allocation across classes (POI/non-POI)", (np.array(labels) == 1).sum()/float(len(labels))
        return
    

# print features

def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    test =  Data_Exploration_Proxy(data_dict) 
    test.run()

if __name__ == '__main__':
    main()