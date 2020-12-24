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





def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1. - x)

def createDropNet(net):
      keep_prob=0.5
      layerdrop=[]    
      for lindex,layer in enumerate(net):
        NeuronDropcache=[]
        for indexn,neuron in enumerate(layer):
            xx=neuron['weights']
            aa=list(xx)
            NeuronDropcache = list(itertools.chain(NeuronDropcache,aa)) 
        layerdrop.append(NeuronDropcache)
# ####
      layerdrop1=[]
      for lindex1,ldrop in enumerate(layerdrop):
        narr=np.array(ldrop)
        NeuronDropcache=[]
        D1 = np.random.uniform(low=-0.5, high=0.5,size=narr.size)
        D1 = D1 < keep_prob
        layerdrop1.append(D1)  
      
      dropnetperneuron=[]
      
      for lindex,layer in enumerate(net):
        layerdrop4=[]
        itr1 = iter(list(layerdrop1[lindex]))
        for neuron in (layer):
            layerdrop3=[]
            for wei in neuron['weights']:
                cc=next(itr1)
                layerdrop3.append(cc) 
            layerdrop4.append(layerdrop3)    
        dropnetperneuron.append(layerdrop4)

      return layerdrop1 ,dropnetperneuron


def ModifiedKendalTau(output,expected,point):   
    tanhrow1=list()
    n=len(expected)  
    for i in range (len(output)):
        sum1=0
        sum1+=(np.tanh(500*(expected[i]-point))*np.tanh(500*(output[i]-point)))         
        tanhrow1.append(2*sum1/(n*(n-1)))
    a=sum(tanhrow1)    
    return a


def DModifiedKendalTau(output,expected,point):   
    tanhrow1=list()
    n=len(expected)
    for i in range (len(output)):
        sum1=0
        x1=((1-np.power(np.tanh(500*(expected[i]-point)),2))*np.tanh(500*(output[i]-point)))
        x2=np.tanh(500*(expected[i]-point))*((1-np.power(np.tanh(500*(output[i]-point)),2)))  
        sum1+=(x1+x2)       
        # tanhrow1.append(2*sum1/(n*(n-1)))
        tanhrow1.append(sum1)
    return tanhrow1 


def initialize_network(features_no):
    input_neurons=features_no
    output_neurons=features_no
    net=list()            
    OneNeuron = [ { 'weights': np.random.uniform(low=-0.1, high=0.1,size=input_neurons)} for i in range(output_neurons) ]
    net.append(OneNeuron)
    return net

def  forward_propagation (net,input,noofclassvalues,scale,dropout,point):
    row=input
    keep_prob=0.5
    cache=[]
    if dropout:
        dropnet,dropnetperneuron=createDropNet(net)
        cache=dropnetperneuron

    for lindex,layer in enumerate(net):
        prev_input=np.array([])
        for indexn,neuron in enumerate(layer):

            xx=neuron['weights']
            nn=len(xx)
            aa=list(xx)
            if dropout:
                D1=dropnet[lindex][indexn*nn:indexn*nn+nn]
                aa = aa * D1    # Shutdown neurons
                aa = aa / keep_prob    # Scales remaining values
            sum = np.array(aa).T.dot(row)
            result=sigmoid(sum)
            neuron['result']=result           
            prev_input=np.append(prev_input,[result])
        row =prev_input   
    return row ,cache

def back_propagation(net,row,expected,noofclassvalues,scale,dropout,cache,point):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results = [neuron['result'] for neuron in net[0]]
                errors =DModifiedKendalTau(results,expected,0.5)
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for inn,neuron in enumerate(nextlayer):
                        if dropout:  
                          zzz=cache[i+1][inn][j]
                          if(zzz):
                              herror+=(neuron['weights'][j]*neuron['delta'])
                        else:      
                          herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                results=[neuron1['result'] for neuron1 in layer]
                # neuron['delta']=errors[j]*DTanhSplitter(neuron['result'],point)
                neuron['delta']=errors[j]*dsigmoid(neuron['result'])

def updateWeights(net,input,lrate,dropout,cache):   
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for inno,neuron in enumerate(net[i]):
            for j in range(len(inputs)):
                if dropout:
                    zzz=cache[i][inno][j]
                    if(zzz):
                        neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
                else:                        
                    neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
            neuron['weights'][-1]+=lrate*neuron['delta']
    return net

def calculateoutputTau(iterationoutput):

        xx=iterationoutput[0]
        vv=iterationoutput[1]
        tau=ModifiedKendalTau(xx,vv,0.5)
        if  np.isnan(tau):
            tau=0
        return tau


def  training(net,X,y, epochs,lrate,n_outputs,noofclassvalues,scale,dropout,point):
    errors=[]
    
    arr1=np.array(X)
    n=arr1.shape[1] #len(X)
    for epoch in range(epochs):     
        sum_Tau1=0
        for col in range(arr1.shape[1]):   #i,row in enumerate(X):     
            xxx1=list(arr1[:, col]) 
            outputs,cache=forward_propagation(net,xxx1,noofclassvalues,scale,dropout,point)
            back_propagation(net,xxx1,y,noofclassvalues,scale,dropout,cache,point)
            net1=updateWeights(net,xxx1,lrate,dropout,cache)
            # print('>epoch=%d,error=%.18f'%(epoch,sum_Tau1))

            sum_Tau1+=calculateoutputTau([y,outputs.tolist()])    
        if epoch%100 ==0:
            print('>-===========epoch=%d,error=%.18f'%(epoch,sum_Tau1))
            errors.append(sum_Tau1)
            # print_network(net)
    return errors ,net1


# Make a prediction with a network# Make a 
def predict(net, row,noofclassvalues,scale,dropout,point):
    outputs,cache = forward_propagation(net, row,noofclassvalues,scale,dropout,point)
    return outputs

###############################################################################################################################

def Test(net1,X_test,y_test,noofclassvalues,scale,point,dropout):
    sum_Tau=0
    iterationoutput=[]
    pred_values=[]
    
    arr1=np.array(X_test)
    n=arr1.shape[1]
    for col in range(n):   #i,row in enumerate(X):
       row=list(arr1[:, col]) 
       pred=predict(net1,np.array(row),noofclassvalues,scale,dropout,point) 
       sum_Tau+=calculateoutputTau([y_test,pred.tolist()])  
    print("Test Predicted rank Error "+"{:.6f}".format(sum_Tau/n)) 
    return sum_Tau/n ,pred.tolist()

def ProcessRoot(net,X,labels,iterations,noofclassvalues,scale,lr,dropout,point ):
    pred_values=[]
    errors,net1=training(net,X,labels,iterations, lr,1,noofclassvalues,scale,dropout,point)
    iterationoutput=[]
    sum_Tau=0
    n=X.shape[1]
    for index in range(X.shape[1]):   #i,row in enumerate(X):
       xxx1=list(X[:, index]) 
       pred=predict(net,np.array(xxx1),noofclassvalues,scale,dropout,point)
       pred_values.append(pred.tolist()[0])
    #    iterationoutput.append([labels,pred.tolist()])
       sum_Tau+=calculateoutputTau([labels,pred.tolist()])  
    print("Predicted Error "+"{:.6f}".format(sum_Tau/n))
    return net1
    # y_actu = pd.Series(pred_values, name='Actual')
    # y_pred = pd.Series(labels[0:100], name='Predicted')

    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    # print (df_confusion)


###############################################################################################################################
def rescale(values,featuresno,data_no, new_min , new_max ):
    totaloutput=[] 
    totaloutput1=[]
    for i in range(featuresno):   #Scle feature
        colvalues=[row[i] for row in values]
        old_min, old_max = min(colvalues), max(colvalues)
        outputf = []
        for v in colvalues:
            new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
            outputf.append(new_v)
        totaloutput.append(outputf)
        ############################ scale instance
    totaloutput2=transpose(totaloutput)      
    for   index, row in enumerate(totaloutput2):
        old_min, old_max = min(row), max(row)
        outputf = []
        for v in row:
            new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
            outputf.append(new_v)
        totaloutput1.append(outputf)
        ############################
    return totaloutput1

def loadData(X,y, featuresno,noofclassvalues, labelno,scale,epoches,lr,dropout,point):
    data = list()
    labels = list()
    alldata = list()
    # print("=================================="+filename+"=============================")
    # filename1 =  filename
    # gpsTrack = open(filename1, "r")
    # csvReader = csv.reader(gpsTrack)
    # next(csvReader)
    # for row in csvReader :
    #         data.append(row[0:featuresno])
    #         labels.append(row[featuresno:featuresno + labelno])
    #         alldata.append(row[:])

    # y = np.array(labels)
    # X = np.array(data)  
    # train_labels = [map(int, i) for i in y]
    # train_features = [map(float, i) for i in X]


    # features_norm=[]
   


    # train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.07, random_state=1)
    # X = np.array([list(item) for item in train_features])
    # y = [list(item) for item in train_labels]
    features_norm = zscore(X, axis=1)
    # y=[x[0] for x in y ]  
    net= initialize_network(len(X))
    net1=ProcessRoot(net,features_norm,y,epoches,noofclassvalues,scale,lr,dropout,point)
    return net1

######################################################################################################
######################################################################################################

# loadData(filename='C:\\Github\\PNN\\Data\\ClassificationData\\glass.csv',featuresno= 9,noofclassvalues=6,labelno=9,scale=5,epoches=50000,lr=0.07,dropout=false,point=3) 


