
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


class Feature_Selection_Proxy:
    def __init__(self,data_dict):
        self.data_dict = data_dict
        return
    def run(self):
        self.__addFractionFeactures() 
        test = data_exploration.Data_Exploration_Proxy(self.data_dict)
        features_list = self.__selectFeatures()
        test.run(features_list)
        return
    def __selectFeatures(self):
        feaDict = {}
        feaDict['1'] = ['poi','bonus','salary']
        feaDict['2'] = ['poi','from_poi_to_this_person','from_this_person_to_poi']
        feaDict['3'] = ['poi','fraction_from_poi','fraction_to_poi','from_this_person_to_poi','from_messages','from_poi_to_this_person','to_messages']
        feaDict['4'] = ['poi','fraction_from_poi','fraction_to_poi']
        return feaDict['1']
    def __computeFraction(self,poi_messages, all_messages ):
        fraction = 0.
        if (poi_messages == 'NaN'or  all_messages == 'NaN'):
            fraction = 0
        else:
            fraction = poi_messages/float(all_messages)
        return fraction
    def __addFractionFeactures(self):
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