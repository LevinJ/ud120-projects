
import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
import numpy as np
import pandas as pd
import data_exploration
from sklearn import tree
from sklearn.feature_selection import SelectKBest,f_classif

class Feature_Selection_Proxy:
    def __init__(self,data_dict):
        self.data_dict = data_dict
        self.data_dict.pop('TOTAL', 0)
        return
    def run(self):
        #add new features
        self.addFractionFeactures() 
        test = data_exploration.Data_Exploration_Proxy(self.data_dict)
        
        #now evaluate existing features
        features_list = self.selectFeatures('7')
        #rank feature importance and reorder
        features_rank =  self.__featureImportance(features_list)
        features_list = features_list[0:1] +  features_rank.tolist()
        #visualize and save selected data
        test.run(features_list)
        
        return
    def selectFeatures(self,sel):
        feaDict = {}
        feaDict['1'] = ['poi','bonus','salary']
        feaDict['2'] = ['poi','from_poi_to_this_person','from_this_person_to_poi']
        feaDict['3'] = ['poi','fraction_from_poi','fraction_to_poi','from_this_person_to_poi','from_messages','from_poi_to_this_person','to_messages']
        feaDict['4'] = ['poi','fraction_from_poi','fraction_to_poi']
        test =  data_exploration.Data_Exploration_Proxy(self.data_dict) 
        feaDict['5'] = test.getAllFeatures()
        feaDict['6'] = ['poi','exercised_stock_options', 'expenses', 'other', 'fraction_to_poi']
        feaDict['7'] = ['poi','exercised_stock_options', 'fraction_to_poi']
        return feaDict[sel]
    def __computeFraction(self,poi_messages, all_messages ):
        fraction = 0.
        if (poi_messages == 'NaN'or  all_messages == 'NaN'):
            fraction = 0
        else:
            fraction = poi_messages/float(all_messages)
        return fraction
    def __featureImportance(self,features_list):
        print "rank feature importance...."
        data = featureFormat(self.data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features = np.array(features)
        features_list1 = self.__decisiointreeImportance(features, labels, features_list)
        features_list2 = self.__anovaImportance(features, labels, features_list)
        
        return features_list2
    def __anovaImportance(self, features, labels, features_list):
        sel = SelectKBest(f_classif, k=2)
        sel.fit(features, labels)
        sortIndexes = sel.scores_.argsort()[::-1]
        features_rank = np.array(features_list[1:])[sortIndexes]
        print "anova f test importance rank: ", features_rank
        return features_rank
    def __decisiointreeImportance(self, features, labels, features_list):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(features, labels)
        sortIndexes = clf.feature_importances_.argsort()[::-1]
        features_rank = np.array(features_list[1:])[sortIndexes]
        print "decision tree importance rank: ", features_rank
        return features_rank
    def addFractionFeactures(self):
        print "add fraction feature to the data dictionary"
        data_dict = self.data_dict
        for name in data_dict:
            data_point = data_dict[name]
            
            from_poi_to_this_person = data_point["from_poi_to_this_person"] 
            to_messages = data_point["to_messages"]
            fraction_from_poi = self.__computeFraction( from_poi_to_this_person, to_messages )
            data_point["fraction_from_poi"] = fraction_from_poi
        
        
            from_this_person_to_poi = data_point["from_this_person_to_poi"] # good features
            from_messages = data_point["from_messages"]
            fraction_to_poi = self.__computeFraction( from_this_person_to_poi, from_messages )
            data_point["fraction_to_poi"] = fraction_to_poi
        return
    
    




def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    test =  Feature_Selection_Proxy(data_dict) 
    test.run()

if __name__ == '__main__':
    main()