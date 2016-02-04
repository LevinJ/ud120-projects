import numpy as np
import itertools 
import tunealgorithm
import pickle
import pandas as pd

class ForwardFeatureSel:
    def __init__(self):
        with open("final_project_dataset.pkl", "r") as data_file:
            data_dict = pickle.load(data_file)
        self.svm =  tunealgorithm.TuneSVM(data_dict)
        self.result = {'f1':[],'precision':[],'recall':[],'best_score':[], 'best_params':[], 'feature_list':[]}
        self.featureList =  ['exercised_stock_options', 'shared_receipt_with_poi', 'expenses',
                             'fraction_to_poi', 'other', 'long_term_incentive', 'total_stock_value',
                             'restricted_stock', 'from_this_person_to_poi', 'from_poi_to_this_person',
                             'bonus', 'director_fees', 'from_messages', 'loan_advances', 'salary',
                             'restricted_stock_deferred', 'fraction_from_poi', 'deferred_income',
                             'total_payments', 'deferral_payments', 'to_messages']
        return
#     def getFeatureCombinaiton(self, numFea):
#         res = [list(x) for x in itertools.combinations(self.featureList, numFea)]
#         return res
    
    def selectBestFeaturList(self, featureLists):
        for fealist in featureLists:
            try:
                fealist.insert(0, 'poi')
                self.svm.setFeatureList(fealist)
                f1, precision,recall, best_score , best_params, feature_list =  self.svm.runGridGridSearchCV() 
                self.result['f1'].append(f1)
                self.result['precision'].append(precision)
                self.result['recall'].append(recall)
                self.result['best_score'].append(best_score)
                self.result['best_params'].append(best_params)
                self.result['feature_list'].append(feature_list)
            except Exception as inst:
                print "XXXXXXX Ignore this combinationXXXXXX", inst, "featues used:  ", fealist
        self.disResult()
        return
    def run(self):
        #round 1 exercised_stock_options, 0.439858   0.762963  0.3090    0.380000 ,{u'kernel': u'rbf', u'C': 5, u'gamma': 120}
        #round 2  exercised_stock_options, fraction_to_poi, 0.419025   0.503153  0.3590    0.375338,{u'kernel': u'rbf', u'C': 45, u'gamma': 200} 
        #round 3  'exercised_stock_options', 'fraction_to_poi', 'from_poi_to_this_person', 0.4718958399491902 0.6466492602262838 0.3715 0.42866666666666675,{'kernel': 'rbf', 'C': 45, 'gamma': 120}
        
        featureLists =  self.generateFeatureList(['exercised_stock_options','fraction_to_poi', 'from_poi_to_this_person'])
        self.selectBestFeaturList(featureLists)
        
        return
    def generateFeatureList(self, basefeature):
        res = []
        for fea in self.featureList:
            if fea in basefeature:
                continue
            tempList = []
            tempList.append(fea)
            res.append(basefeature + tempList)
        return res
    def disResult(self):
        df = pd.DataFrame(self.result, columns= ['f1','precision','recall','best_score', 'best_params', 'feature_list' ])
        df2 = df.sort(columns = ['f1'], ascending=False)
        print df2
        print "@@@@@@@best combinations: ", df2[0:1].values
        return
    
def main():
    test = ForwardFeatureSel() 
#     res = test.generateFeatureList([])
#     print res
#     res = test.generateFeatureList(['exercised_stock_options'])
#     print res
#     
#     res = test.generateFeatureList(['exercised_stock_options'])
#     print res
#     
#     res = test.generateFeatureList(['exercised_stock_options', 'expenses'])
#     print res
    
    test.run()
#     d = {'one':[2,3,1,5,5],
#          'two':[5,4,3,2,7],
#          'letter':['a','a','b','b','c']}
#     df = pd.DataFrame(d, index = d['two'], columns= ['one', 'two','letter'])
#     test = df.sort(columns= ['one'], ascending=False)
# #     test = df.sort(columns= ['one', 'two'], ascending=False)
#     print test

if __name__ == '__main__':
    main()