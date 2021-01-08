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
from PNN import PNN

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

def loadData():
    data = list()
    print("==================================Hand Writing=============================")
    data = pd.read_csv("C:\\Github\\PNN\\Data\\Images\\A_Z Handwritten Data.csv").astype('float32')
    X = data.drop('0',axis = 1)
    y = data['0'] 
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
    return train_x.values, test_x.values, train_y.values, test_y.values
  
X,X_test,y,y_test= loadData()   
cutval=len(X)
X=X[0:cutval]
X_test=X_test[0:cutval]
y=y[0:cutval]
y_test=y_test[0:cutval]
##############################################Building Tree 3 models#####################################
# print(X[0])
pnn = PNN()
X123,y123=removeDataByLabelList(X,y,[0,1,2])
y123b=fuzzy3Labels(y123)
net1,trainedlabels1= pnn.loadData(X=X123,y=y123b, featuresno= 784, labelno=3,labelvalue=3, iteration=100,lrate=0.07,hn=2,scale=30)

# for i in range(len(X)):
# # plt.subplot(3,3,i+1)
#   if(y[i]==5 or y[i]==8):
#     plt.imshow(X[i].reshape(28,28),cmap='gray')
#     plt.show()
# plt.show()
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

data1,labels1,data2,labels2=splitData(X,y,12)
yb=labelBinarySplitter(y,12)

data11,labels11,data12,labels12=splitData(data1,labels1,18)

print("============ Rank from 18 to 25 ")
yb1=labelBinarySplitter(labels11,21)
net1,trainedlabels1= pnn.loadData(X=data11,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=2,scale=30)


print("============ Rank from 12 to 18 ")
yb1=labelBinarySplitter(labels12,15)
net1,trainedlabels1= pnn.loadData(X=data12,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=2,scale=30)



data21,labels21,data22,labels22=splitData(data2,labels2,6)
print("============ Rank from 6 to 12 ")
yb1=labelBinarySplitter(labels21,9)
net1,trainedlabels1= pnn.loadData(X=data21,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=2,scale=30)

print("============ Rank from 0 to 6 ")
yb1=labelBinarySplitter(labels22,3)
net1,trainedlabels1= pnn.loadData(X=data22,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=100,lrate=0.07,hn=2,scale=30)


X123,y123=removeDataByLabelList(X,y,[0,1,2])
y123b=fuzzy3Labels(y123)
net1,trainedlabels1= pnn.loadData(X=X123,y=y123b, featuresno= 784, labelno=3,labelvalue=3, iteration=100,lrate=0.07,hn=2,scale=30)

# net1,trainedlabels1=PreferenceNeuron.loadData(X=X,y=yb,featuresno= 784,noofclassvalues=2,labelno=2,scale=30,epoches=500,lr=0.07,dropout=true) 

# X2,y2,X22,y22=categorDataByBinaryResult(X,y,trainedlabels)

X2,y2=removeDataByLabelList(X,y,[0,1])
X22,y22=removeDataByLabelList(X,y,[2,3])
X222,y222=removeDataByLabelList(X,y,[4,5])
X2222,y2222=removeDataByLabelList(X,y,[6,7])
X22222,y22222=removeDataByLabelList(X,y,[8,9])

print("===========================Ranking [5,6] ==========================================")
yb1=binaryLabels(y2)
net1,trainedlabels1=pnn.loadData(X=X2,y=yb1, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)

# X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)
print("===========================Ranking [3,4] ==========================================")
yb2=binaryLabels(y22)
net1,trainedlabels1=pnn.loadData(X=X22,y=yb2, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)

# X_1,y_1,X_11,y_11=trainTestingSplitter(X222,y222)
print("===========================Ranking [1,2] ==========================================")
yb3=binaryLabels(y222)
net1,trainedlabels1=pnn.loadData(X=X222,y=yb3, featuresno= 784, labelno=2,labelvalue=2, iteration=500,lrate=0.07,hn=5,recurrent=False,scale=30)

##############################################################################################
##############################################################################################
###################################Testing the 3 models#######################################

# X_1,y_1,X_11,y_11=trainTestingSplitter(X2,y2)

# y_11b=binarySplitter(y_11,4)
# rooterror,pred_values=PreferenceNeuron.Test(net1,X_11,y_11b,noofclassvalues=2,scale=5,subrank=2,dropout=False)

# X_test2,y_test2,X_test3,y_test3=categorDataByBinaryResult(X_11,y,pred_values)

# rootresult=ClassifierNeuron.Test(net2,X_test2,y_test2,steps=5,startindex=0,scale=5,dropout=False)
# rootresult=ClassifierNeuron.Test(net3,X_test3,y_test3,steps=5,startindex=5,scale=5,dropout=False)