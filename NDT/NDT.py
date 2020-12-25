import numpy as np
# from IPython.display import Image,display
import matplotlib.pyplot as plt

# Use Python 3.7.3
import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
from scipy.stats import zscore
import scipy.stats
from statistics import mean 
from itertools import combinations, permutations
import csv
import scipy.stats as ss
from sympy import *
import random
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import itertools
import numpy.ma as ma
from scipy._lib.six import iteritems
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime
import mnist
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import PartialRankerNeuron
import ClassifierNeuron

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)

###############################################################################################################################
def removeDataByLabelIndex(X,y,labelIndex):
    outputData=[]
    outputLabels=[]
    for indexl,row in enumerate(y):
      if row!=labelIndex:
         outputData.append(X[indexl])
         outputLabels.append(row)
    return  outputData , outputLabels

def removeDataByLabelList(X,y,labelList): 
    outputData=[]
    outputLabels=[]
    for indexl,row in enumerate(y):
       for labelIndex in (labelList): 
         if row==labelIndex:
            outputData.append(X[indexl])
            outputLabels.append(row)
    return  outputData , outputLabels




def splitData(X_data,labels,classno):
    list1=[]
    list2=[] 
    outputdata1=[]
    outputdata2=[]
    for index, i in enumerate(labels):
        if i >classno:
            list1.append(i)
            outputdata1.append(X_data[index])
        else:
            list2.append(i)
            outputdata2.append(X_data[index])

    return outputdata1,list1,outputdata2,list2


def splitterData(train_features,train_labels):

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.3, stratify=train_labels,random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X,y,X1,y1

def categoryLabels(labels):
    newlist=[]
    for lab in labels:
        if(lab>4):
           newlist.extend([3])
        elif(lab<5 and lab>2):
           newlist.extend([2])
        else:
           newlist.extend([1])   

        # if(lab>3):
        #    newlist.extend([2])
        # else:
        #    newlist.extend([1])       
    return newlist



def categorResult(originalData,trainedlabels):
    newlabel1=[]
    newlabel2=[]
    newlabel3=[]
    newdata1=[]
    newdata2=[]
    newdata3=[]
    for index,lab in enumerate(trainedlabels):
        if(round(lab)==1):
           newlabel1.append(round(lab))
           newdata1.append(originalData[index])
        elif(round(lab)==2):
           newlabel2.append(round(lab))
           newdata2.append(originalData[index])
        else:
           newlabel3.append(round(lab)) 
           newdata3.append(originalData[index])  

    return newdata1 ,newlabel1,newdata2,newlabel2,newdata3,newlabel3


def loadData(filename, featuresno, labelno,labelvalues):
    data = list()
    labels = list()
    alldata = list()
    print("=================================="+filename+"=============================")
    filename1 =  filename
    gpsTrack = open(filename1, "r")
    csvReader = csv.reader(gpsTrack)

    next(csvReader)
    for row in csvReader :
            data.append(row[0:featuresno])
            labels.append(row[featuresno:featuresno + labelno])
            alldata.append(row[:])

    y = np.array(labels)
    X = np.array(data)  
 
    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(X, y,stratify = y, test_size=0.3, random_state=1)
    
    train_labels = [map(float, i) for i in train_labels]
    train_features = [map(float, i) for i in train_features]

    test_features = [map(float, i) for i in test_features]
    test_labels = [map(float, i) for i in test_labels]

    X = np.array([list(item) for item in train_features])
    y = np.array([list(item) for item in train_labels])
    X1 = np.array([list(item) for item in test_features])
    y1 = np.array([list(item) for item in test_labels])
    y=[g[0] for g in y ] 
    y1=[g[0] for g in y1 ]

    return X,y,X1,y1

 
filename='C:\\Github\\PNN\\Data\\ClassificationData\\glass.csv'
X,y,X1,y1 = loadData(filename, featuresno=9,labelno=1,labelvalues=6) 
##############################################Building Tree 3 models#####################################


yb=categoryLabels(y)


net1,trainedlabels=PartialRankerNeuron.loadData(X=X,y=yb,featuresno= 9,noofclassvalues=3,labelno=9,scale=30,epoches=1000,lr=0.07,dropout=false) 


X2,y2,X22,y22,X33,y33=categorResult(X,trainedlabels)

# X2,y2=removeDataByLabelList(X,y,[1,2])
# X22,y22=removeDataByLabelList(X,y,[3,4])
# X33,y33=removeDataByLabelList(X,y,[5,6])

X_1,y_1,X_1,y_1=splitterData(X2,y2)
net2=ClassifierNeuron.loadData(X_1,y_1,X_1,y_1,featuresno= 9,steps=3,startindex=1,noofclassvalues=2,labelno=1,scale=10,epoches=5000,lr=0.05,dropout=false) 

X_2,y_2,X_2,y_2=splitterData(X22,y22)
net3=ClassifierNeuron.loadData(X_2,y_2,X_2,y_2,featuresno= 9,steps=3,startindex=3,noofclassvalues=2,labelno=1,scale=5,epoches=1000,lr=0.05,dropout=false) 

X_3,y_3,X_3,y_3=splitterData(X33,y33)
net3=ClassifierNeuron.loadData(X_3,y_3,X_3,y_3,featuresno= 9,steps=3,startindex=5,noofclassvalues=2,labelno=1,scale=5,epoches=1000,lr=0.05,dropout=false) 

##############################################################################################
###################################Testing the 3 models#######################################

X_test=X[:,0:3]
y_train=categoryLabels(y)
y_test=categoryLabels(y)
rooterror,pred_values=PartialRankerNeuron.Test(net1,X_test,y_test,noofclassvalues=3,scale=5,point=3,dropout=False)

X_test2,y_test2,X_test3,y_test3 =splitData(X,y,6)

rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=3,startindex=1,scale=5,dropout=False)
rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=3,startindex=4,scale=5,dropout=False)