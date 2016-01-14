#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
from sklearn import preprocessing




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


def isNaN(num):
    return num == "NaN"
def findLargestOutlier():
    min_exercised_stock_options=0
    max_exercised_stock_options = 0
    min_salary=5e10
    max_salary=0
    for key, value in data_dict.iteritems():
        if not (isNaN(value['exercised_stock_options'])):
            if min_exercised_stock_options == 0:
                min_exercised_stock_options = value['exercised_stock_options']
            if (value['exercised_stock_options'] < min_exercised_stock_options):
                min_exercised_stock_options = value['exercised_stock_options']
            if max_exercised_stock_options == 0:
                max_exercised_stock_options = value['exercised_stock_options']
            if (value['exercised_stock_options'] > max_exercised_stock_options):
                max_exercised_stock_options = value['exercised_stock_options']
        if not isNaN(value['salary']):       
            if (value['salary'] < min_salary):
                min_salary = value['salary']
            if (value['salary'] > max_salary):
                max_salary = value['salary']
    print "min_exercised_stock_options", min_exercised_stock_options           
    print "max_exercised_stock_options", max_exercised_stock_options
    print "min_salary", min_salary
    print "max_salary", max_salary
    return

findLargestOutlier()
### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
# features_list = [poi, feature_1, feature_2, "total_payments"]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


nparr = np.array(finance_features)
df = pd.DataFrame({'salary':nparr[:,0], 'exercised_stock_options':nparr[:,1]})
print df.describe()
### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
# for f1, f2,f3 in finance_features:
nparr = nparr.astype(np.float64)
min_max_scaler = preprocessing.MinMaxScaler()
finance_features = min_max_scaler.fit_transform(nparr)
temparr = np.array([[200000.0, 1000000.0]])
print min_max_scaler.transform(temparr)

for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
#plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred


cluster_alg = KMeans(n_clusters=2, random_state=42)

cluster_alg.fit(finance_features)
pred = cluster_alg.predict(finance_features)



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
