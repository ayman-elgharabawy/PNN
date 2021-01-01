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
import ClassifierNN
import PNN

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


def trainTestingSplitter(train_features,train_labels):

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels,shuffle=true, test_size=0.2, random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X,y,X1,y1

def binaryLabels(labels):
    newlist=[]
    for lab in labels:
        if( lab>=4):
           newlist.append([2,1])
        else:
           newlist.append([1,2])   
    return newlist


def categorRankingResult(originalData,y,trainedlabels):
    newlabel1=[]
    newlabel2=[]

    newdata1=[]
    newdata2=[]

    for index,lab in enumerate(originalData):
        if(round(trainedlabels[index,0])==2):
           newlabel1.append(y[index])
           newdata1.append(originalData[index])
        else:
           newlabel2.append(y[index])
           newdata2.append(originalData[index])  

    return newdata1 ,newlabel1,newdata2,newlabel2


def binarySplitter(labels,splitpoint):
    newlist=[]
    for lab in labels:
        if( lab>splitpoint):
           newlist.append([2,1])
        else:
           newlist.append([1,2])   
    return newlist

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
 
    y=[(float)(g[0]) for g in y ] 
    # X=[(float)(g) for g in X ] 
    X =[[float(y) for y in x] for x in X]
    # y1=[g[0] for g in y1 ]

    return X,y #,X1,y1

 
filename='C:\\Github\\PNN\\Data\\ClassificationData\\glass.csv'
XX,yy = loadData(filename, featuresno=9,labelno=1,labelvalues=6) 
##############################################Building Tree 3 models#####################################

yb=binaryLabels(yy)

net1,trainedlabels1=PNN.loadData(X=XX,y=yb, featuresno= 9, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=1,recurrent=False,scale=30)
X_1,y_1,X_11,y_11=trainTestingSplitter(XX,yy)

yb1=binaryLabels(y_11)


# X2,y2,X22,y22=categorRankingResult(X,y,trainedlabels)

X2,y2=removeDataByLabelList(XX,yy,[4,5,6])
X22,y22=removeDataByLabelList(XX,yy,[1,2,3])

X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)

# label2=binarySplitter(y2,5)

net2=ClassifierNN.loadData(X_1,y_1,X_11,y_11,featuresno= 9,steps=3,startindex=4,noofclassvalues=3,scale=5,epoches=5000,hn=5,lr=0.07,dropout=false) 
# net1,trainedlabels1=PNN.loadData(X=X2,y=label2, featuresno= 9, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=1,recurrent=False,scale=30)


X_1,y_1,X_11,y_11=trainTestingSplitter(X22,y22)
net3=ClassifierNN.loadData(X_1,y_1,X_11,y_11,featuresno= 9,steps=3,startindex=1,noofclassvalues=3,scale=5,epoches=5000,hn=5,lr=0.07,dropout=false) 


label22=binarySplitter(y22,2)
net1,trainedlabels1=PNN.loadData(X=X22,y=label22, featuresno= 9, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=1,recurrent=False,scale=30)



X_2,y_2,X_2,y_2=trainTestingSplitter(X22,y22)


##############################################################################################
###################################Testing the 3 models#######################################

y_1b=binaryLabels(y1)


X_test2,y_test2,X_test3,y_test3=categorRankingResult(X_11,y_11,pred_values)

rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=3,startindex=4,scale=5,dropout=False)
rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=3,startindex=1,scale=5,dropout=False)