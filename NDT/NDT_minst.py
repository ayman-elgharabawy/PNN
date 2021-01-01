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
import PreferenceNeuron
import ClassifierNeuron
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


def trainTestingSplitter(train_features,train_labels):

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.3,random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X,y,X1,y1



def binarySplitter(labels,splitpoint):
    newlist=[]
    for lab in labels:
        if( lab>splitpoint):
           newlist.append([2,1])
        else:
           newlist.append([1,2])   
    return newlist


def categorDataByBinaryResult(originalData,y,trainedlabels):
    newlabel1=[]
    newlabel2=[]

    newdata1=[]
    newdata2=[]

    for index,lab in enumerate(originalData):
        if(round(trainedlabels[index])==2):
           newlabel1.append(y[index])
           newdata1.append(originalData[index])
        else:
           newlabel2.append(y[index])
           newdata2.append(originalData[index])  

    return newdata1 ,newlabel1,newdata2,newlabel2

def convertMultilabel(train_labels,labelno):
    newLabels=np.array([[0]*labelno]*len(train_labels))

    for index,row in enumerate(train_labels):
         newLabels[index,row]=1

    return newLabels

def loadData():
    data = list()
    labels = list()
    alldata = list()
    print("==================================Minsit=============================")
    train_images = mnist.train_images()[:100]
    train_labels = mnist.train_labels()[:100]
    test_images = mnist.test_images()[:100]
    test_labels = mnist.test_labels()[:100]
    flatterd=[]
    for tup in train_images:   
       flatterd.append(tup.ravel() )

    y = np.array(train_labels)
    X = np.array(flatterd)  
 
    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(X, y,stratify = y, test_size=0.3, random_state=1)
    return train_features, test_features, train_labels, test_labels 

 

X,X1,y,y1 = loadData() 
##############################################Building Tree 3 models#####################################
print(len(X[0]))
yb=binarySplitter(y,4)
net1,trainedlabels1= PNN.loadData(X=X,y=yb, featuresno= 784, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=5,recurrent=False,scale=30)

# net1,trainedlabels1=PreferenceNeuron.loadData(X=X,y=yb,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 

# X2,y2,X22,y22=categorDataByBinaryResult(X,y,trainedlabels)

X2,y2=removeDataByLabelList(X,y,[0,1,2,3,4])
X22,y22=removeDataByLabelList(X,y,[5,6,7,8,9])


y2_m=binarySplitter(y2,2)
net1,trainedlabels2=PreferenceNeuron.loadData(X=X2,y=y2_m,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 

y22_m=binarySplitter(y22,7)
net1,trainedlabels3=PreferenceNeuron.loadData(X=X22,y=y22_m,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 


X3,y3=removeDataByLabelList(X,y,[3,4])
X33,y33=removeDataByLabelList(X,y,[0,1,2])

y3_m=binarySplitter(y3,3)
net1,trainedlabels3=PreferenceNeuron.loadData(X=X3,y=y3_m,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 

y33_m=binarySplitter(y33,1)
net1,trainedlabels3=PreferenceNeuron.loadData(X=X33,y=y33_m,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 


X4,y4=removeDataByLabelList(X,y,[8,9])
X44,y44=removeDataByLabelList(X,y,[5,6,7])


X_1,y_1,X_1,y_1=trainTestingSplitter(X44,y44)
net2=ClassifierNeuron.loadData(X_1,y_1,X_1,y_1,featuresno= 784,steps=2,startindex=6,noofclassvalues=5,scale=5,epoches=5000,lr=0.05,dropout=false) 

X_2,y_2,X_2,y_2=trainTestingSplitter(X33,y33)
net3=ClassifierNeuron.loadData(X_2,y_2,X_2,y_2,featuresno= 784,steps=2,startindex=1,noofclassvalues=5,scale=5,epoches=5000,lr=0.05,dropout=false) 
##############################################################################################
##############################################################################################
###################################Testing the 3 models#######################################

X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)

y_11b=binarySplitter(y_11,4)
rooterror,pred_values=PreferenceNeuron.Test(net1,X_11,y_11b,noofclassvalues=2,scale=5,subrank=2,dropout=False)

X_test2,y_test2,X_test3,y_test3=categorDataByBinaryResult(X_11,y,pred_values)

rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=5,startindex=0,scale=5,dropout=False)
rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=5,startindex=5,scale=5,dropout=False)