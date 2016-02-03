import sys
import pickle
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.grid_search import GridSearchCV
import tester
from sklearn.cross_validation import StratifiedShuffleSplit

sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit

def dispEvalResult(predictions, labels_test):
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



with open("../../final_project/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
features_list = ['poi','exercised_stock_options', 'expenses']
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features = np.array(features)


scaler = min_max_scaler = preprocessing.MinMaxScaler()
# features = scaler.fit_transform(features)

tuned_parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [10, 100, 1000,5000,10000], 'kernel': ['rbf']},
 ]
score = 'recall'


min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=StratifiedShuffleSplit(labels, 1000, random_state = 42),
                       scoring='f1')
# clf = SVC(kernel='rbf',C=100, gamma=10000)
# clf = Pipeline([('scaler', min_max_scaler), ('svc', clf)])

clf.fit(features, labels)
print clf.best_params_
print clf.best_score_
tester.test_classifier(clf.best_estimator_, labels, features)