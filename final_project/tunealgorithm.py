
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

class Tune_Algorithm_Proxy(feature_selection.Feature_Selection_Proxy):
    def run(self):
        self.addFractionFeactures()
        features_list = self.selectFeatures('7')
        clf = tree.DecisionTreeClassifier(min_samples_split=20)
        self.runTrain(clf,features_list)
        self.runTest(clf,features_list)
        return
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
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        PERF_FORMAT_STRING = "\
        \tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
        Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
        RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
        \tFalse negatives: {:4d}\tTrue negatives: {:4d}"
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
        return
    def runTest(self,clf,features_list):
        dump_classifier_and_data(clf, self.data_dict, features_list)
        tester.main()
        return

    


def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    test =  Tune_Algorithm_Proxy(data_dict) 
    test.run()

if __name__ == '__main__':
    main()

