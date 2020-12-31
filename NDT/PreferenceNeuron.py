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
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



def Spearman(output,expected):
    n=len(expected)
    dif=0
    diflist=np.array([])
    for i in range (n):
        diflist=np.append(diflist,[np.power(output[i]-expected[i],2)])
        dif+=np.power(output[i]-expected[i],2)

    den=n*(np.power(n,2)-1)
    deflist=np.array([])
    for dd in diflist:
      deflist=np.append(deflist,((6*dd)/(den)))
    return deflist 

def DSpearman(output,expected):
    n=len(expected)
    dif=0
    diflist=np.array([])
    deflist=np.array([])
    for i in range (n):
        diflist=np.append(diflist,[2*(output[i])-2*(expected[i])])
        dif+=2*(output[i])-2*(expected[i])
      
    den=n*(np.power(n,2)-1)
    for dd in diflist:
        deflist=np.append(deflist,[((dd)/(den))])
    return deflist  

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

def initialize_network(features_no):
    input_neurons=features_no
    output_neurons=2
    net=list()            
    OneNeuron = [ { 'weights': np.random.uniform(low=-0.01, high=0.01,size=input_neurons)} for i in range(output_neurons) ]
    net.append(OneNeuron)
    return net

def  forward_propagation (net,input,noofclassvalues,scale,dropout):
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
            result=SSS(sum,noofclassvalues,scale)
            neuron['result']=result           
            prev_input=np.append(prev_input,[result])
        row =prev_input   
    return row ,cache

def back_propagation(net,row,expected,noofclassvalues,scale,dropout,cache):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results = [neuron['result'] for neuron in net[0]]
                errors =DSpearman(results,expected)
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
                neuron['delta']=errors[j]*SSS(neuron['result'],noofclassvalues,scale)

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
                        neuron['weights'][j]-=lrate*neuron['delta']*inputs[j]
                else:                        
                    neuron['weights'][j]-=lrate*neuron['delta']*inputs[j]
            neuron['weights'][-1]-=lrate*neuron['delta']
    return net

def calculateoutputTau(iterationoutput):

        tau,pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])  
        if  np.isnan(tau):
            tau=0
        return tau


def  training(net,X,y, epochs,lrate,n_outputs,noofclassvalues,scale,dropout):
    errors=0 
    arr1=np.array(X)
    n=len(X)
    for epoch in range(epochs):     
        sum_Tau1=0
        for ind,xxx1 in enumerate(X):   #i,row in enumerate(X):     
            outputs,cache=forward_propagation(net,xxx1,noofclassvalues,scale,dropout)
            back_propagation(net,xxx1,y[ind],noofclassvalues,scale,dropout,cache)
            net1=updateWeights(net,xxx1,lrate,dropout,cache)
            sum_Tau1+=calculateoutputTau([y[ind],outputs.tolist()])    
        # if epoch%10 ==0:
        #     print('>-===========epoch=%d,error=%.18f'%(epoch,sum_Tau1/n))
        errors+=(sum_Tau1/n)

    return (errors/epochs) ,net1


# Make a prediction with a network# Make a 
def predict(net1, row,noofclassvalues,scale,dropout):
    outputs,cache = forward_propagation(net1, row,noofclassvalues,scale,dropout)
    return outputs

###############################################################################################################################

def Test(net1,X_test,y_test,noofclassvalues,scale,subrank,dropout):
    sum_Tau=0
    pred_values=[]
    n=len(X_test)
    for i,row in enumerate(X_test):    #i,row in enumerate(X):
       pred=predict(net1,np.array(row),noofclassvalues,scale,dropout) 
       pred_values.append(pred.tolist()[0])
       sum_Tau+=calculateoutputTau([y_test[i],pred.tolist()]) 
       print("Test Predicted rank Error "+"{:.6f}".format(sum_Tau/(i+1)))  
    print("Test Predicted rank Error "+"{:.6f}".format(sum_Tau/n)) 
    return sum_Tau/n ,pred_values

def ProcessRoot(net1,X,labels,iterations,noofclassvalues,scale,lr,dropout ):
    pred_values=[]
    kfold = KFold(5, True, 1)
    avrError=0
    for idx_train, idx_test in kfold.split(X):
        errors,net2=training(net1,X[idx_train,:],labels,iterations, lr,1,noofclassvalues,scale,dropout)
        sum_Tau=0
        n=len(X[idx_test,:])
        for i,row in enumerate(X[idx_test,:]):  
            pred=predict(net2,np.array(row),noofclassvalues,scale,dropout)
            pred_values.append(pred.tolist()[0])
            sum_Tau+=calculateoutputTau([labels[idx_test[i]],pred.tolist()])
        print("Predicted Fold Error "+"{:.8f}".format(sum_Tau/n))
        avrError+=(sum_Tau/n) 
    print("Avr. 10 folds Error "+"{:.6f}".format(avrError/10))

    return net2 ,pred_values


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

def loadData(X,y, featuresno,noofclassvalues, labelno,scale,epoches,lr,dropout):

    #features_norm=rescale(X,featuresno,len(X), -scale , scale )
    features_norm = zscore(X, axis=1)
    xx=len(X[1])
    net= initialize_network(xx)
    net1,traineddata=ProcessRoot(net,features_norm,y,epoches,noofclassvalues,scale,lr,dropout)
    return net1 ,traineddata

######################################################################################################
######################################################################################################

# loadData(filename='C:\\Github\\PNN\\Data\\ClassificationData\\glass.csv',featuresno= 9,noofclassvalues=6,labelno=9,scale=5,epoches=50000,lr=0.07,dropout=false,point=3) 


