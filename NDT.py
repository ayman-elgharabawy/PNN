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

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#StairStep SS Function#
def SSS(xi,nlabel,bx):
    sum2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        sum2 +=0.5*(np.tanh((-b*(xi))-(xx)))
    sum2=-1*sum2     
    sum2= sum2+(nlabel*0.5)  
    return sum2      

#StairStep SS Function Derivative#
def dSSS(xi,nlabel,bx): 
    derivative2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        derivative2 +=0.5*(1-np.power(np.tanh((-b*(xi))-(xx)),2))
    derivative2=-1*derivative2     
    derivative2= derivative2+(nlabel*0.5)  
    return derivative2

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


def initialize_network():
    input_neurons=len(X[0])
    output_neurons=1
    net=list()       
    OneNeuron = [ { 'weights': np.random.uniform(low=-0.1, high=0.1,size=input_neurons)} for i in range(output_neurons) ]
    net.append(OneNeuron)
    return net
# def initialize_network():
#     input_neurons=len(X[0])
#     hidden_neurons=input_neurons*4
#     output_neurons=1
#     n_hidden_layers=1
#     net=list()
#     for h in range(n_hidden_layers):
#         if h!=0:
#             input_neurons=len(net[-1])
            
#         hidden_layer = [ { 'weights': np.random.uniform(low=-0.1, high=0.1,size=input_neurons)} for i in range(hidden_neurons) ]
#         net.append(hidden_layer)
    
#     output_layer = [ { 'weights': np.random.uniform(low=-0.1, high=0.1,size=hidden_neurons)} for i in range(output_neurons)]
#     net.append(output_layer) 
#     return net

def  forward_propagation (net,input):
    row=input
    for index, layer in enumerate(net):
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)
            result=SSS(sum,3,1)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input   
    return row

def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = (expected-np.array(results))/1000 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                results=[neuron1['result'] for neuron1 in layer]
                neuron['delta']=errors[j]*dSSS(neuron['result'],3,1)


def updateWeights(net,input,lrate):   
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
            neuron['weights'][-1]+=lrate*neuron['delta']

def  training(net,X,y, epochs,lrate,n_outputs):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs=forward_propagation(net,row)
            sum_error+=math.sqrt(math.pow(y[i]-outputs,2)) 
            back_propagation(net,row,y[i])
            updateWeights(net,row,0.05)
        if epoch%10 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
            # print_network(net)
    return errors


# Make a prediction with a network# Make a 
def predict(net, row):
    outputs = forward_propagation(net, row)
    return outputs

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

###############################################################################################################################
def removeDataByLabelIndex(X,y,labelIndex):
    outputData=[]
    outputLabels=[]
    for indexl,row in enumerate(y):
      if row!=labelIndex:
         outputData.append(X[indexl])
         outputLabels.append(row)
    return  outputData , outputLabels

def ProcessRoot(X,labels,iterations ):
    pred_error=0
    pred_values=[]
    errors=training(net,X,labels,iterations, 0.07,1)
    for index,y in enumerate(X[0:100]):
       pred=predict(net,np.array(y))
       pred_values.append(pred.tolist()[0])
       pred_error+=math.sqrt(math.pow(labels[index]-pred,2))
    print("Predicted Error "+str(pred_error))



    y_actu = pd.Series(pred_values, name='Actual')
    y_pred = pd.Series(labels[0:100], name='Predicted')

    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print (df_confusion)
    # matrix = classification_report(y_actu,y_pred,labels=[2,1,0])
    # print('Classification report : \n',matrix)

    # plot_confusion_matrix(df_confusion)

###############################################################################################################################


def loadData(filename, featuresno, labelno):
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
    train_labels = [map(int, i) for i in y]
    train_features = [map(float, i) for i in X]

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.2, random_state=1)
    X = np.array([list(item) for item in train_features])
    y = [list(item) for item in train_labels]

    y=[x[0] for x in y ]  
    return X,y

 

X,y = loadData('C:\\ayman\\PhDThesis\\iris.csv', 4,1) 
net=initialize_network()
ProcessRoot(X,y,500)
X1,y1=removeDataByLabelIndex(X,y,0)
ProcessRoot(X1,y1,500)

# print_network(net)

# Confusion matrix

# y_actu = pd.Series([2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2], name='Actual')
# y_pred = pd.Series([0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2], name='Predicted')

# df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
# print (df_confusion)
# matrix = classification_report(y_actu,y_pred,labels=[1,0])
# print('Classification report : \n',matrix)

# plot_confusion_matrix(df_confusion)
