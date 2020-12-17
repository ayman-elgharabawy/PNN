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



def Spearman(output,expected):
    n=len(expected)
    nem=0
    for i in range (n):
        nem+=np.power(output[i]-expected[i],2) 
    den=n*(np.power(n,2)-1)
    bb=1-((6*nem)/(den)) 
    if bb>100:
        print("Weired.. "+bb)
    if(np.isnan(bb)):
        bb=0
    return bb

#StairStep SS Function#

def SSS(xi,nlabel,start,bx):
   
    sum2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        sum2 +=0.5*(np.tanh((-b*(xi))-(xx)))
    sum2=-1*sum2  
    sum2= sum2+(start+(nlabel/2))
    return sum2   

def dSSS(xi,nlabel,start,bx):
    derivative2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        derivative2 +=0.5*(1-np.power(np.tanh((-b*(xi))-(xx)),2))
    derivative2=-1*derivative2     
    derivative2= derivative2+(start+(nlabel/2))  
    return derivative2



def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)



    


# Make a prediction with a network# Make a 
def predict(net, row,steps,startindex,scale,Afunction):
    outputs = forward_propagation(net, row,steps,startindex,scale,Afunction)
    return outputs



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


def ProcessRoot(net,X,labels,X1,labels1,iterations,loop ,steps,startindex,lrate,scale,Afunction):
    pred_error=0
    pred_values=[]
    errors,net=training(net,X,labels,iterations,lrate,1,steps,startindex,scale,Afunction)
    for index,y in enumerate(X1):
       pred=predict(net,np.array(y),steps,startindex,scale,Afunction)
       pred_values.append(pred.tolist()[0])
       if(Afunction=='SS'):
         pred_error+=math.sqrt(math.pow(labels1[index]-pred,2))
    if(Afunction=='SS'):     
       print("Predicted Error "+str(pred_error))
    else:
        print("Predicted Ranking Error "+str(ss.spearmanr(pred_values,labels1)))
    # y_actu = pd.Series(pred_values, name='Actual')
    # y_pred = pd.Series(labels1, name='Predicted')

    # df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'])
    # print (df_confusion)
    # finalpredictedvalues=[]
    # finalIndex=0
    # for index,k in enumerate(df_confusion.columns):
    #     counter1=0
    #     counter2=0
    #     for i in range(len(df_confusion.values)):
    #         counter2+=df_confusion.values[i,index]
    #         if(round(df_confusion.index[i])==k):
    #             finalpredictedvalues.append(round(df_confusion.index[i]))
    #             counter1+= df_confusion.values[i,index]
    #     if(counter1==counter2):
    #         print("K="+str(k))
    #         finalIndex= k

    # finalpredictedvalues=np.unique(finalpredictedvalues) 
    # counter=0
    # for t in (df_confusion.columns):
    #     for r in finalpredictedvalues: 
    #         if(t==round(r)):
    #               counter+=1 
    #               break
    # if counter!= len(df_confusion.columns) :  
    #     print("Increasing no. of iterations = "+str(iterations+2000))             
    #     ProcessRoot(net,X,labels,X1,labels1,iterations+2000,loop,steps,startindex,lrate,scale)          
    # if (finalIndex!=0):
    #     return finalIndex
    # if loop==7:
    #     return 0 
    # else:
    #     print("Increasing no. of iterations = "+str(iterations+2000))             
    #     ProcessRoot(net,X,labels,X1,labels1,iterations+2000,loop,steps,startindex,lrate,scale)        

    return net
    # matrix = classification_report(y_actu,y_pred,labels=[2,1,0])
    # print('Classification report : \n',matrix)

    # plot_confusion_matrix(df_confusion)

###############################################################################################################################

def testData(net,X1,labels1,steps,startindex,scale,Afunction):
    pred_values=[]
    pred_error=0
    for index,y in enumerate(X1):
       pred=predict(net,np.array(y),steps,startindex,scale,Afunction)
       pred_values.append(pred.tolist()[0])
       if(Afunction=='SS'):
          pred_error+=math.sqrt(math.pow(labels1[index]-pred,2))
    if Afunction=='SS':
       print("Predicted Test Error "+str(pred_error))
    else:
       print("Predicted Ranking Error "+str(ss.spearmanr(pred_values,labels1)))
    
    return pred_values

def splitterData(train_features,train_labels):

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.3, stratify=train_labels,random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X,y,X1,y1

def categoryLabels(labels,noofclasses):
    newlist=[]
    for lab in labels:
        if(lab>=(round(noofclasses/2))):
           newlist.extend([2])
        else:
           newlist.extend([1])
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

 

X,y,X1,y1 = loadData('C:\\Github\\PNN\\Data\\ClassificationData\\glass.csv', featuresno=9,labelno=1,labelvalues=6) 
##############################################Building Tree 3 models#####################################

net=initialize_network()
yb=categoryLabels(y,6)
yb1=categoryLabels(y1,6)

net1=ProcessRoot(net,X,yb,X1,yb1,10000,0,steps=2,startindex=1,lrate=0.07,scale=5,Afunction='Splitter')

X2,y2=removeDataByLabelList(X,y,[1,2,3])
X22,y22=removeDataByLabelList(X,y,[4,5,6])

X4,y4,X3,y3=splitterData(X2,y2)
net2=ProcessRoot(net,X4,y4,X3,y3,10000,0,steps=3,startindex=1,lrate=0.07,scale=5,Afunction='SS')


X4,y4,X3,y3=splitterData(X22,y22)
net3=ProcessRoot(net,X4,y4,X3,y3,10000,0,steps=3,startindex=4,lrate=0.07,scale=5,Afunction='SS')


##############################################################################################
###################################Testing the 3 models#######################################

X_train, X_test,y_train , y_test  =sklearn.model_selection.train_test_split(X, y,stratify = y, test_size=0.3, random_state=1)
y_train=categoryLabels(y_train,6)
y_test=categoryLabels(y_test,6)
rootresult=testData(net1,X_test,y_test,steps=2,startindex=1,scale=5,Afunction='Splitter')
X_test2,y_test2=removeDataByLabelList(X_train,rootresult,[1,2,3])
X_test3,y_test3=removeDataByLabelList(X_train,rootresult,[4,5,6])

rootresult=testData(net2,X_test2,y_test2,steps=3,startindex=1,scale=5,Afunction='SS')
rootresult=testData(net3,X_test3,y_test3,steps=3,startindex=4,scale=5,Afunction='SS')