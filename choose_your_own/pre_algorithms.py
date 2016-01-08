from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def getAdaboost():
    print "Adaboost used"
    dt_stump = DecisionTreeClassifier(max_depth=1)
    return AdaBoostClassifier(base_estimator=dt_stump,n_estimators=150,random_state=78)

def getSVM():
    print "SVM used"
    return SVC(kernel='rbf', gamma=500)

def getTree():
    print "Decision tree used"
    return DecisionTreeClassifier(min_samples_leaf=10)

def getNaiveBayes():
    print "NaiveBayes used"
    return GaussianNB()

options = {0 : getAdaboost,
           1 : getSVM,
           2 : getTree,
           3 : getNaiveBayes,
           }
def getClf():
    return options[0]()

