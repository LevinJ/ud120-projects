#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

def isNaN(num):
    return num == "NaN"


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
num_poi = 0
num_salary = 0
num_knownemailaddress=0
num_item = 0
num_total_payments = 0
poi_nanpayment = 0
for key, value in enron_data.iteritems():
    num_item = num_item + 1
    if value["poi"] == True:
        num_poi = num_poi + 1
        if isNaN(value['total_payments']):
            poi_nanpayment = poi_nanpayment +1
        
#         print key
    if not isNaN(value['salary']):
        num_salary = num_salary +1
    else:
        print "unkonw salary", key
    if not isNaN(value['email_address']):
        num_knownemailaddress = num_knownemailaddress + 1
    else:
        print "unknown email", key
    if not isNaN(value['total_payments']):
        num_total_payments = num_total_payments + 1
    else:
        print "uknown total_payments", key
        

print "num_item ", num_item
print "num_salary ", num_salary
print "num_knownemailaddress ", num_knownemailaddress        
print "number of poi ", num_poi
print "unkowntotal paymetns %f, percentage %f" %(num_item-num_total_payments, float((num_item-num_total_payments))/num_item)
print "poi nan payments %f, percentage %f" %(poi_nanpayment, float(poi_nanpayment)/num_poi)


