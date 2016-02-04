
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
import feature_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import  easytester
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

class Tune_Algorithm_Proxy(feature_selection.Feature_Selection_Proxy):
    def __init__(self,data_dict):
        feature_selection.Feature_Selection_Proxy.__init__(self, data_dict)
        self.addFractionFeactures()
        return
    def setFeatureList(self, selected_features):
        self.selected_features = selected_features
        return
    def run(self):
        print "#########################Tune on algorithm: ", self.getAlgName(), "######################3"
        self.addFractionFeactures()
        features_list = self.getFeatureList()
        clf = self.getClf()
        self.runTrain(clf,features_list)
        self.runTest(clf,features_list)
        return
#     def getClf(self):
#         return
#     def getFeatureList(self):
#         return
    def runGridGridSearchCV(self):
        print "######################### runGridGridSearchCV: ", self.getAlgName(), "######################3"
        features_list = self.getFeatureList()
        print "Test feature list: ", features_list
        clf = self.getClf(usepipeine = False)
        data = featureFormat(self.data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features = np.array(features)
        min_max_scaler = preprocessing.MinMaxScaler()
        features = min_max_scaler.fit_transform(features)
        # do grid search
        clf = GridSearchCV(clf, self.getTunedParamterOptions(), cv=StratifiedShuffleSplit(labels, 100, random_state = 42),
                       scoring='f1')
        clf.fit(features, labels)
        print clf.best_params_
        print clf.best_score_ 
        f1,precision,recall = easytester.test_classifier(clf.best_estimator_, labels, features)
        return f1,precision,recall, clf.best_score_ , clf.best_params_, features_list
    def runTrain(self,clf,features_list):
        data = featureFormat(self.data_dict, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features = np.array(features)
        clf.fit(features, labels)
        predictions = clf.predict(features)
        print 'test result on training data....'
        self.dispEvalResult(predictions, labels)
        return
    def dispEvalResult(self,predictions, labels_test):
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
        \tFalse negatives: {:4d}\tTrue negatives: {:4d}"
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        PERF_FORMAT_STRING = "\
        \tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
        Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print ""
        return
    def runTest(self,clf,features_list):
        print "test result on stratified cross validation data...."
        dump_classifier_and_data(clf, self.data_dict, features_list)
        tester.main()
        return

    
class TuneDecisionTree(Tune_Algorithm_Proxy):
    def getClf(self,usepipeine = True):
        return tree.DecisionTreeClassifier(min_samples_split=1)
    def getFeatureList(self):
        if hasattr(self, 'selected_features'):
            return self.selected_features
        return self.selectFeatures('7')
    def getTunedParamterOptions(self):
        tuned_parameters = [
          {'min_samples_split': [1, 2,3, 4, 5, 6,7,8,9]},
         ]
        return tuned_parameters
    def getAlgName(self):
        return "Decision Tree"
    
class TuneNavieBayes(Tune_Algorithm_Proxy):
    def getClf(self):
        return GaussianNB()
    def getFeatureList(self):
        return self.selectFeatures('7')
    def getAlgName(self):
        return "Navie Bayes"
    
class TuneSVM(Tune_Algorithm_Proxy):
    def getClf(self,usepipeine = True):
        clf = SVC(kernel='rbf',C=45, gamma=200)
        if not usepipeine:
            return clf
        min_max_scaler = preprocessing.MinMaxScaler()
        clf = Pipeline([('scaler', min_max_scaler), ('svc', clf)])
        return clf
    def getTunedParamterOptions(self):
        tuned_parameters = [
          {'C': [1, 5, 15, 45,100,500,1000,1500], 'gamma': [120, 180, 200,240,1000,1500,3000,5000,7000,9000,10000], 'kernel': ['rbf']},
         ]
        return tuned_parameters
    def getFeatureList(self):
        if hasattr(self, 'selected_features'):
            return self.selected_features
        return self.selectFeatures('7')
    def getAlgName(self):
        return "Support Vector Machine"

def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
#     dt =  TuneDecisionTree(data_dict) 
#     dt.run()
#     dt.runGridGridSearchCV()
#     
#     nb =  TuneNavieBayes(data_dict) 
#     nb.run() 
    
    svm =  TuneSVM(data_dict) 
    svm.setFeatureList(['poi','exercised_stock_options', 'fraction_to_poi'])
#     svm.run()
    svm.runGridGridSearchCV() 

if __name__ == '__main__':
    main()

