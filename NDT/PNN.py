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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime

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
    
def print_network(net,epoch,tau,row1):
    # with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
        print("------------------------------------------------------ Epoch "+str(epoch)+" ---------------------------------------\n")
        print("Input row:" +str(row1))  
        for i, layer in enumerate(net, 1):
            if (i==1):
                print("=============== Middle layer =============== \n")           
            else:
                print("=============== Output layer =============== \n")      
            for j, neuron in enumerate(layer, 1):
                print("Neuron {} :".format(j), neuron)  
        print("==== Roh Correlation = "+str(tau)+"======\n")       
      
def initialize_network(ins, hiddens, outs):
    
    input_neurons = ins
    hidden_neurons = hiddens
    output_neurons = outs

    net = list()

    # for h in range(n_hidden_layers):
    # if h != 0:
    # input_neurons = len(net[-1])
    hidden_layer = {'middle':[ {'weights': np.random.uniform(low=-0.5, high=0.5,size=input_neurons)} for i in range(hidden_neurons)] }
    net.append(hidden_layer)
    output_layer = {'output':[{'weights': np.random.uniform(low=-0.5, high=0.5,size=hidden_neurons)} for i in range(output_neurons)] }   
    net.append(output_layer)

    return net 


def forward_propagation(net, input1,trainfold, n_outputs,labelvalue,b,statelayer,recurrent):

    row1 = input1
    prev_input = np.array([])
    for nindex,neuron in enumerate(net[0]['middle']):
        if(recurrent):
           sum1 = neuron['weights'].T.dot(row1)+statelayer[nindex]
        else:
           sum1 = neuron['weights'].T.dot(row1)
        result = SSS(sum1,labelvalue,b)  
        if math.isnan(result):
              result=0    
        neuron['result'] = result 
        prev_input = np.append(prev_input, [result])    
    row1 = prev_input 
    statelayer=prev_input 
    
    prev_input = np.array([])
    for neuron in net[1]['output']:
        sum1 = neuron['weights'].T.dot(row1)
        result = SSS(sum1,labelvalue,b)   
        if math.isnan(result):
              result=0   
        neuron['result'] = result 
        prev_input = np.append(prev_input, [result])
  
    row1 = prev_input 

    #############################
    # ###########################  
    return row1 ,statelayer


def updateNeuronResults(layer,row1):

    for indexn,neuron in enumerate(layer):
         neuron['result']=row1[indexn]

def back_propagation(net, row, expected, outputs, n_outputs,labelvalue,b):

    results = list()
    errors = np.array([]) #list()#np.zeros(n_outputs)
    results = [neuron['result'] for neuron in net[1]['output']]
    errors =DSpearman(results,expected)

    for indexsub,neuron in enumerate(net[1]['output']):   
        neuron['delta'] =errors[indexsub] * dSSS(neuron['result'],labelvalue,b)

    for indexsub,neuron in enumerate(net[0]['middle']):
        herror = 0
        for j,outneuron in enumerate(net[1]['output']):
            herror += (neuron['weights'][j]) * (outneuron['delta'])   
        errors=np.append(errors,[herror])

    for indexsub,neuron in enumerate(net[0]['middle']):
        neuron['delta'] =errors[indexsub] * dSSS(neuron['result'],labelvalue,b)


def updateWeights(net, input1, lrate):

    inputs = list(input1)
    for neuron in (net[0]['middle']):
        for j in range(len(inputs)):
            neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] -= lrate * neuron['delta']    
    inputs = [neuron['result'] for neuron in net[0]['middle']]
    for neuron in (net[1]['output']):
        for j in range(len(inputs)):
            neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] -= lrate * neuron['delta']    

    return net,neuron

def PNNFit(net,epochs,train_fold_features,train_fold_labels,n_outputs,labelvalue,lrate,hn,b,recurrent):
    iterationoutput=np.array([])
    vv=np.array([])
    statelayer=[0]*hn 
    for epoch in range(epochs):
        for i, row in enumerate((train_fold_features)):
            xxx1=np.array(list(row))       
            trainfoldexpected=list(train_fold_labels[i])
            outputs ,statelayer = forward_propagation(net, xxx1,trainfoldexpected, n_outputs,labelvalue,b,statelayer,recurrent)
            back_propagation(net, xxx1, trainfoldexpected, outputs, n_outputs,labelvalue,b)
            net,neuron=updateWeights(net, xxx1, lrate) 
        net=net     
        iterationoutput=np.append(iterationoutput,calculateoutputTau([outputs,np.array(trainfoldexpected)]))
    net=net    
    z=len(train_fold_features)    
    rr=sum(iterationoutput/z)    
    return net,rr 

def calculateoutputTau(iterationoutput):
        sum_Tau=0
        tau,pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])   
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

def truncate(n, decimals=0):
    if  np.isnan(n):
        return 0
    multiplier = 100 ** decimals
    v=int(n * multiplier) / multiplier
    return v


def CrossValidationAvg(kfold,foldcounter,foldindex,X_train,y_train,featuresno, noofhidden, labelno,labelvalue,lrate,bbs,epochs,bestvector,recurrent):
    net = initialize_network(featuresno, noofhidden, labelno)
    avr_res=0
    tot_etau=0
    for idx_train, idx_test in kfold.split(X_train):      
        foldindex += 1
        trainlabel=[]
        testlabel=[]      
        for i in idx_train:
           trainlabel.append(y_train[i]) 
        for i in idx_test:
           testlabel.append(y_train[i])   
        net,error=PNNFit(net,epochs,X_train[idx_train,:],trainlabel,labelno,labelvalue,lrate,noofhidden,bbs,recurrent)
        iterationoutput=predict(net,X_train[idx_test,:],testlabel,  labelno,labelvalue,bbs,noofhidden,recurrent)
        print("-- Predition one fold Result %d",iterationoutput)
        tot_etau+=iterationoutput
    avr_res=tot_etau/foldcounter  
    print("Final average %f Folds test Result %f",foldcounter,avr_res)
        

    return avr_res, net
  

def training(epochs, X,y, featuresno, labelno,labelvalue,lrate,hn,scale,recurrent):
    foldcounter=5
    kfold = KFold(foldcounter, True, 1)
    foldindex = 0
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    lrlist=[0.05]#,0.09,0.1,0.2,0.3,0.4,0.5]
    scalelist=[2]
    hnlist=[hn]
    bestvector=[0,0,0,0,0]
    avresult=0
    bestvresult=0
    for hn1 in hnlist: 
        for lr1 in lrlist:
            for scl in scalelist:
                avresult,bestnet=CrossValidationAvg(kfold,foldcounter,foldindex,X,y,featuresno, hn1, labelno,labelvalue,lr1,scl,epochs,bestvector,recurrent)
                print('crossv Prediction=%f , lr=%f',(avresult,lr1))
                if(avresult>bestvresult):
                    bestvresult=avresult
                    bestvector=[bestnet,lr1,hn1,bestvresult,scl]
    now = datetime.now()
    timestamp = datetime.timestamp(now)

    # with open(Datasetfilename+str(timestamp)+'.txt', 'a') as f:
    # print(">>>>>>>>>>>>>>>>>>>>>>>>"+Datasetfilename+"<<<<<<"+str(timestamp)+"<<<<<<<<<<<<<<<<<<<<<")
    # print(">>>>>>>>Best Parameters<<<<<<<<<")
    # print(">>>>>>>>Best Vector Data<<<<<<<<<")
    # print('scale=%f,best result=%f',(bestvector[4],bestvresult))
    # print(">>>>>>>>Testing data result<<<<<<<<<")
    # X_test_norm = zscore(X_test, axis=0)
    # iterationoutput=predict(bestnet,X_test_norm, y_test, labelno,labelvalue,bestvector[4],hn1,recurrent)
    # print('Final Prediction=%f',iterationoutput)
    print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return bestnet,avresult


def predict(net1,test_fold_features,test_fold_labels, n_outputs,labelvalue,bx,hn,recurent):
    iterationoutput=np.array([])
    statelayer=list()
    for i, row in enumerate((test_fold_features)): 
        xxx1=np.array(list(row))    
        testfoldlabels=list(test_fold_labels[i])
        predicted,statelayer= forward_propagation(net1, row,testfoldlabels, n_outputs,labelvalue,bx,statelayer,recurent)
        iterationoutput=np.append(iterationoutput,[calculateoutputTau([predicted,np.array(testfoldlabels)])])      
    avrre=sum(iterationoutput)/len(test_fold_features) 
    return  avrre



def loadData(X,y, featuresno, labelno,labelvalue, iteration,lrate,hn,recurrent,scale):
   
    features_norm = zscore(X, axis=1)

    net1,tot_error2=training( iteration, features_norm,y, featuresno, labelno,labelvalue,lrate,hn,scale,recurrent)
                                      
    print('Done')
    return net1
 ###############################################################

###############################################################################################################################
###############################################################################################################################
##################################################################################################################################

