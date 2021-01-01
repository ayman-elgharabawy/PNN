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
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



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
        D1 = np.random.uniform(low=-0.9, high=0.9,size=narr.size)
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


#StairStep SS Function#
def SSS(xi,nlabel,start,bx):
    sum2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel-1):
        xx=s-((i*t)/(nlabel-1))
        sum2 +=0.5*(np.tanh((-b*(xi))-(xx)))
    sum2=-1*sum2     
    sum2= sum2+(nlabel*0.5)-0.5 +start 
    return sum2     

#StairStep SS Function Derivative#
def dSSS(xi,nlabel,start,bx): 

    derivative2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel-1):
        xx=s-((i*t)/(nlabel-1))
        derivative2 +=0.5*(1-np.power(np.tanh((-b*(xi))-(xx)),2))
    derivative2=-1*derivative2     
    derivative2= derivative2+(nlabel*0.5)-0.5 +start 
    return derivative2

def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


# def initialize_network(features_norm):
#     input_neurons=len(features_norm[0])
#     output_neurons=1
#     net=list()       
#     OneNeuron = [ { 'weights': np.random.uniform(low=-0.1, high=0.1,size=input_neurons)} for i in range(output_neurons) ]
#     net.append(OneNeuron)
#     return net

def initialize_network(ins, hiddens, outs):
    
    input_neurons = ins
    hidden_neurons = hiddens
    output_neurons = outs
    net = list()

    hidden_layer = {'middle':[ {'weights': np.random.uniform(low=-0.9, high=0.9,size=input_neurons)} for i in range(hidden_neurons)] }
    net.append(hidden_layer)

    output_layer = {'output':[{'weights': np.random.uniform(low=-0.9, high=0.9,size=hidden_neurons)} for i in range(output_neurons)] }   
    net.append(output_layer)

    return net 


def  forward_propagation (net,input,steps,startindex,scale,dropout):
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
            result=SSS(sum,steps,startindex,scale)
            neuron['result']=result           
            prev_input=np.append(prev_input,[result])
        row =prev_input   
    return row ,cache

def back_propagation(net,row,expected,nlabel,start,scale,dropout,cache):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = (expected-np.array(results))/100
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
                neuron['delta']=errors[j]*dSSS(neuron['result'],nlabel,start,scale)


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

def  training(net,X,y, epochs,steps,startindex,lrate,n_outputs,noofclassvalues,scale,dropout):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        outlist=[]
        for i,row in enumerate(X):
            outputs,cache=forward_propagation(net,row,steps,startindex,scale,dropout)
            outlist.append(outputs)
            back_propagation(net,row,y[i],steps,startindex,scale,dropout,cache)
            net=updateWeights(net,row,lrate,dropout,cache)

        if epoch%100 ==0:
            mse = mean_squared_error(y, outlist)
            rmse = sqrt(mse)
            print('>epoch=%d,RMS error=%.9f'%(epoch,rmse))
            # errors.append(sum_error)
            # print_network(net)
    return rmse , net


# Make a prediction with a network# Make a 
def predict(net, row,steps,startindex,scale,dropout):
    outputs,cache = forward_propagation(net, row,steps,startindex,scale,dropout)
    return outputs

###############################################################################################################################

def Test(net1,X_test,y_test,steps,startindex,scale,dropout=False):
    pred_error=0
    pred_values=[]
    for index,row in enumerate(X_test):
       pred=predict(net1,np.array(row),steps,startindex,scale,dropout)
       pred_values.extend(pred)
    #    pred_error+=math.sqrt(math.pow(y_test[index]-pred,2)) 
    mse = mean_squared_error(y_test, pred_values)
    rmse = sqrt(mse)   
    print("Test Classifier Predicted Error "+str(rmse))   
    print("==================================================")
    y_actu = pd.Series(pred_values, name='Actual')
    y_pred = pd.Series(y_test, name='Predicted')

    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print (df_confusion)
    return rmse


def ProcessRoot(net,X,labels,iterations,steps,startindex,noofclassvalues,scale,lr,dropout ):
    # pred_error=0
    pred_values=[]
    errors,net1=training(net,X,labels,iterations,steps,startindex, lr,1,noofclassvalues,scale,dropout)
    for index,y in enumerate(X):
       pred=predict(net,np.array(y),steps,startindex,scale,dropout)
       pred_values.append(pred)
    #    pred_error+=math.sqrt(math.pow(labels[index]-pred,2))
    mse = mean_squared_error(labels, pred_values)
    rmse = sqrt(mse)   
    print("Predicted Error "+str(rmse))

    # y_actu = pd.Series(pred_values, name='Actual')
    # y_pred = pd.Series(labels[0:100], name='Predicted')
    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    # print (df_confusion)

    # matrix = classification_report(y_actu,y_pred,labels=[2,1,0])
    # print('Classification report : \n',matrix)

    # plot_confusion_matrix(df_confusion)
    return net1
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

def loadData(filename, featuresno,noofclassvalues,scale,epoches,hn,lr,dropout):
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
            labels.append(row[featuresno:featuresno + 1])
            alldata.append(row[:])

    y = np.array(labels)
    X = np.array(data)  
    train_labels = [map(int, i) for i in y]
    train_features = [map(float, i) for i in X]

    features_norm=[]
    features_norm_all=[]

    train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(train_features, train_labels, test_size=0.2, random_state=1)
    X = np.array([list(item) for item in train_features])
    y = [list(item) for item in train_labels]
     
    data_no=len(X)
    features_norm = zscore(X, axis=1)
    # features_norm = rescale(X,featuresno,data_no,-scale,scale) 

    y=[x[0] for x in y ]  
    net=initialize_network(ins=features_norm,hiddens=hn, outs=1)
    net1=ProcessRoot(net,features_norm,y,epoches,noofclassvalues,scale,lr,dropout)
    return net1

def loadData(X,y,X_test,y_test,featuresno,steps,startindex,noofclassvalues,scale,epoches,hn,lr,dropout):

    net=initialize_network(ins=len(X[0]),hiddens=hn,outs=1)
    net1=ProcessRoot(net,X,y,epoches,steps,startindex,noofclassvalues,scale,lr,dropout)
    return net1
######################################################################################################
######################################################################################################


