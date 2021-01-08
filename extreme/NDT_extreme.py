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
import ExtremePNN
from ExtremePNN import ExtremePNN
from xclib.data import data_utils

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
    labels1=[]
    labels2=[] 
    outputdata1=[]
    outputdata2=[]
    for index, i in enumerate(labels):
        if i >classno:
            labels1.append(i)
            outputdata1.append(X_data[index])
        else:
            labels2.append(i)
            outputdata2.append(X_data[index])

    return outputdata1,labels1,outputdata2,labels2


def trainTestingSplitter(train_features,train_labels):

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.3,random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X,y,X1,y1



def labelBinarySplitter(labels,splitpoint):
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

def fuzzy3Labels(labels):
    newlist=[]
    for lab in labels:
        if (lab>4):
            newlist.append([1,2,3])
        elif(lab>2 and lab<=4) :
            if (lab==3):
               newlist.append([3,1,2])
            elif(lab==4):
               newlist.append([2,1,3]) 
        else:
            newlist.append([3,2,1])      
    return newlist

def fuzzy5Labels(labels):
    newlist=[]
    for lab in labels:
        print(lab)
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

def loadData(fileName):
    
    # from sklearn.datasets import fetch_rcv1
    # rcv1 = fetch_rcv1()

    # ndarray = rcv1.data
    # features = ndarray.tolist()
    
    # ndarray1 = rcv1.target.toarray()
    # labels = ndarray1.tolist()

    # return rcv1.data,rcv1.target
    features_csr, tabels_csr, num_samples, num_features, num_labels = data_utils.read_data(fileName,header=True)
    ndarray = features_csr.toarray()
    features = ndarray.tolist()
    ndarray = tabels_csr.toarray()
    labels = ndarray.tolist()
    return features,labels

    
#############################################s#####################################
# #############################################s#####################################
# #############################################s#####################################
# #############################################s#####################################  
X,y= loadData('C:\\Github\\PNN\\Data\\extreme\\Delicious\\Delicious\\Delicious_data.txt')   #//featuresno= 500, labelno=983
# X,y= loadData()   #featuresno= 120, labelno=101


pnn = ExtremePNN()
# X123,y123=removeDataByLabelList(X,y,[0,1,2])
# y123b=fuzzy3Labels(y123)
net1,trainedlabels1= pnn.loadData(X=X[0:500],y=y[0:500], featuresno= 500 , labelno=983,labelvalue=2, iteration=5,lrate=0.07,hn=2,scale=1)


