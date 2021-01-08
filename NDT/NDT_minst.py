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
import PNN
<<<<<<< HEAD
from PNN import PNN

=======
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3

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


def fuzzy5Labels(labels):
    newlist=[]
    # ranking = ctrl.Antecedent(np.arange(0, 6, 1), 'ranking')
    # ranking.automf(3)
    # ranking['low'] = fuzz.trimf(ranking.universe, [0, 0, 2])
    # ranking['medium'] = fuzz.trimf(ranking.universe, [0, 13, 25])
    # ranking['high'] = fuzz.trimf(ranking.universe, [13, 25, 25])
    for lab in labels:
        if (lab>7):
            newlist.append([1,2,3,4,5])
        elif(lab>5 and lab<=7) :
            if (lab==7):
               newlist.append([2,1,3,4,5])
            elif(lab==6):
               newlist.append([3,1,2,4,5]) 
        elif(lab>3 and lab<=5) :
            if (lab==5):
               newlist.append([5,2,1,3,4])
            elif(lab==4):
               newlist.append([5,3,1,2,4]) 
        elif(lab>1 and lab<=3) : 
            if (lab==3):
               newlist.append([5,4,2,1,3])
            elif(lab==2):
               newlist.append([5,4,3,1,2]) 
        elif(lab<=1) :
            newlist.append([5,4,3,2,1])
            
   
    return newlist

def binaryLabels(labels):
    newlist=[]
    for lab in labels:
        if( lab>=4):
           newlist.append([2,1])
        else:
           newlist.append([1,2])   
    return newlist

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
    return X,y
    # train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(X, y,stratify = y, test_size=0.3, random_state=1)
    # return train_features, test_features, train_labels, test_labels 

 

X,y = loadData() 
##############################################Building Tree 3 models#####################################
print(len(X[0]))
# yb=binarySplitter(y,4)
<<<<<<< HEAD
pnn = PNN()
yb=fuzzy5Labels(y)

net1,trainedlabels1= pnn.loadData(X=X,y=yb, featuresno= 784, labelno=5,labelvalue=5, iteration=100,lrate=0.07,hn=20,scale=30)
=======

yb=fuzzy5Labels(y)

net1,trainedlabels1= PNN.loadData(X=X,y=yb, featuresno= 784, labelno=5,labelvalue=5, iteration=100,lrate=0.07,hn=20,recurrent=False,scale=30)
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3

# net1,trainedlabels1=PreferenceNeuron.loadData(X=X,y=yb,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 

# X2,y2,X22,y22=categorDataByBinaryResult(X,y,trainedlabels)

X2,y2=removeDataByLabelList(X,y,[0,1])
X22,y22=removeDataByLabelList(X,y,[2,3])
X222,y222=removeDataByLabelList(X,y,[4,5])
X2222,y2222=removeDataByLabelList(X,y,[6,7])
X22222,y22222=removeDataByLabelList(X,y,[8,9])

print("===========================Ranking [5,6] ==========================================")
yb1=binaryLabels(y2)
<<<<<<< HEAD

net1,trainedlabels1=pnn.loadData(X=X2,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,scale=30)
=======
net1,trainedlabels1=PNN.loadData(X=X2,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3

# X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)
print("===========================Ranking [3,4] ==========================================")
yb2=binaryLabels(y22)
<<<<<<< HEAD
net1,trainedlabels1=pnn.loadData(X=X22,y=yb2, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,scale=30)
=======
net1,trainedlabels1=PNN.loadData(X=X22,y=yb2, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3

# X_1,y_1,X_11,y_11=trainTestingSplitter(X222,y222)
print("===========================Ranking [1,2] ==========================================")
yb3=binaryLabels(y222)
<<<<<<< HEAD
net1,trainedlabels1=pnn.loadData(X=X222,y=yb3, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,scale=30)
=======
net1,trainedlabels1=PNN.loadData(X=X222,y=yb3, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3

##############################################################################################
##############################################################################################
###################################Testing the 3 models#######################################

<<<<<<< HEAD
# X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)

# y_11b=binarySplitter(y_11,4)
# rooterror,pred_values=PreferenceNeuron.Test(net1,X_11,y_11b,noofclassvalues=2,scale=5,subrank=2,dropout=False)

# X_test2,y_test2,X_test3,y_test3=categorDataByBinaryResult(X_11,y,pred_values)

# rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=5,startindex=0,scale=5,dropout=False)
# rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=5,startindex=5,scale=5,dropout=False)
=======
X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)

y_11b=binarySplitter(y_11,4)
rooterror,pred_values=PreferenceNeuron.Test(net1,X_11,y_11b,noofclassvalues=2,scale=5,subrank=2,dropout=False)

X_test2,y_test2,X_test3,y_test3=categorDataByBinaryResult(X_11,y,pred_values)

rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=5,startindex=0,scale=5,dropout=False)
rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=5,startindex=5,scale=5,dropout=False)
>>>>>>> 2e9c48c051b1439132267b33e2d1090f18a805e3
