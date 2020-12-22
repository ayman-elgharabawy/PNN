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
import mnist
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

import skimage.data  


# conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
# pool = MaxPool2()                  # 26x26x8 -> 13x13x8
# softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10
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



def SpearmanDistance(output,expected):
    n=len(expected)
    dif=0
    diflist=np.array([])
    for i in range (n):
        diflist=np.append(diflist,[np.power(output[i]-expected[i],2)])
        dif+=np.power(output[i]-expected[i],2)
    deflist=np.array([])
    for dd in diflist:
      deflist=np.append(deflist,(0.5*dd))
    return deflist[0]

def DSpearmanDistance(output,expected):
    n=len(expected)
    dif=0
    diflist=np.array([])
    for i in range (n):
        diflist=np.append(diflist,[(output[i])-(expected[i])])
        dif+=(output[i])-(expected[i])
    deflist=np.array([]) 
    for dd in diflist:
        deflist=np.append(deflist,[(dd)])
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
      
def initialize_network(ins, hiddens, outs, n_hlayers):
    
    input_neurons = ins
    hidden_neurons = hiddens
    output_neurons = outs
    n_hidden_layers = n_hlayers
    net = list()

    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])
        hidden_layer = {'middle':[ {'weights': np.random.uniform(low=-0.9, high=0.9,size=input_neurons)} for i in range(hidden_neurons)] }
        # state_layer ={'state': [{'weights': np.random.uniform(low=-0.9, high=0.9,size=input_neurons)} for i in range(hidden_neurons)] }
        net.append(hidden_layer)
        # net.append(state_layer)
    output_layer = {'output':[{'weights': np.random.uniform(low=-0.9, high=0.9,size=hidden_neurons)} for i in range(output_neurons)] }   
    net.append(output_layer)

    return net 


def forward_propagation(net, input1,trainfold, n_outputs,labelvalue,b,hn,statelayer,recurrent):
    
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

def back_propagation(net, row, expected, outputs, n_outputs,labelvalue,datalength,b):

    results = list()
    errors = np.array([]) #list()#np.zeros(n_outputs)
    results = [neuron['result'] for neuron in net[1]['output']]
    results.append(0)
    errors =DSpearmanDistance(results,[expected,0])

    for indexsub,neuron in enumerate(net[1]['output']):   
        neuron['delta'] =errors[indexsub] * dSSS(neuron['result'],labelvalue,b)

    for indexsub,neuron in enumerate(net[0]['middle']):
        herror = 0
        for j,neuron in enumerate(net[1]['output']):
            herror += (neuron['weights'][j]) * (neuron['delta'])   
        errors=np.append(errors,[herror])

    for indexsub,neuron in enumerate(net[0]['middle']):
        neuron['delta'] =errors[indexsub] * dSSS(neuron['result'],labelvalue,b)


def updateWeights(net, input1, lrate):

    inputs = list(input1)
    for neuron in (net[0]['middle']):
        for j in range(len(inputs)):
            neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] += lrate * neuron['delta']    
    inputs = [neuron['result'] for neuron in net[0]['middle']]
    for neuron in (net[1]['output']):
        for j in range(len(inputs)):
            neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] += lrate * neuron['delta']    

    return neuron

def PNNFit(net,train_fold_features,train_fold_labels,n_outputs,labelvalue,lrate,epoch,datalength,hn,b,recurrent):
    iterationoutput=np.array([])
    vv=np.array([])
    statelayer=[0]*hn 
    for i, row in enumerate((train_fold_features)):
        xxx1=np.array(list(row))       
        trainfoldexpected=train_fold_labels[i]
        outputs ,statelayer = forward_propagation(net, xxx1,trainfoldexpected, n_outputs,labelvalue,b,hn,statelayer,recurrent)
        back_propagation(net, xxx1, trainfoldexpected, outputs, n_outputs,labelvalue,datalength,b)
        updateWeights(net, xxx1, lrate)  
        iterationoutput=np.append(iterationoutput,calculateoutputTau([outputs,np.array(trainfoldexpected)]))
    z=len(train_fold_features)    
    rr=sum(iterationoutput/z)    
    return rr 

def calculateoutputTau(iterationoutput):
        sum_Tau=0
        # tau,pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])  
        tau=SpearmanDistance([iterationoutput[1]], iterationoutput[0]) 
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

def truncate(n, decimals=0):
    if  np.isnan(n):
        return 0
    multiplier = 100 ** decimals
    v=int(n * multiplier) / multiplier
    return v

def trainingNoValidation(epochs, n_outputs, train_features,train_labels, featuresno, labelno,labelvalue,lrate,hn,b,recurrent):
    net = initialize_network(featuresno, hn, n_outputs,1)
    errorRate = np.array([])
    sum_Tau=np.array([])
    n=len(train_features)
    for epoch in range(epochs):     
          iterationoutput=PNNFit(net,train_features,train_labels,n_outputs,labelvalue,lrate,epoch,n,hn,b,recurrent)
          sum_Tau=np.append(sum_Tau,iterationoutput) 
          if epoch % 10 == 0:
            print('Training >epoch=%d,Tau=%.4f' % (epoch, sum(sum_Tau)/(epoch+1)))
            errorRate=np.append(errorRate,[sum_Tau/(epoch+1)])
    Terror=sum(sum_Tau)/epochs        
    print('Final training result Tau=%.4f ' %(Terror)    )
    print("Iteration end.")   
    return Terror


def CrossValidationAvg(kfold,foldindex,n,foldederrorrate,X_train,y_train,featuresno, noofhidden, labelno,labelvalue,lrate,bbs,epochs,bestvector,Datasetfilename,trainingdataresult,recurrent):
    net = initialize_network(featuresno, noofhidden, labelno,1)
    avr_res=0
    for idx_train, idx_test in kfold.split(X_train):      
        foldindex += 1
        train_fold_features=X_train[idx_train,:]
        train_fold_features_norm = zscore(train_fold_features, axis=0)
        train_fold_labels=y_train[idx_train,:]
        test_fold_features=X_train[idx_test,:]
        test_fold_features_norm = zscore(test_fold_features, axis=0)
        test_fold_labels=y_train[idx_test,:]
        tot_etau=np.array([])
        errorRate_validate=np.array([])
        sum_Tau=np.array([])
        for epoch in range(epochs):
            iterationoutput_train=PNNFit(net,train_fold_features_norm,train_fold_labels,labelno,labelvalue,lrate,epoch,n,noofhidden,bbs,recurrent)
            # print("Train Epoch %d, %f",(epoch,iterationoutput_train))
            sum_Tau=np.append(sum_Tau,iterationoutput_train) 
            iterationoutput=predict(test_fold_features_norm, test_fold_labels, net, labelno,labelvalue,bbs,noofhidden,recurrent)
            tot_etau=np.append(tot_etau,[iterationoutput])
            # print("-- Predition Result %d",iterationoutput)
        avr_res=sum(tot_etau)/(epochs)  
        print("Final average Predition Result %f",avr_res)
        

    return avr_res, net
  

def training(Datasetfilename,epochs, alldata, featuresno, labelno,labelvalue,lrate,hn,scale,trainingdataresult,recurrent):
    kfold = KFold(10, True, 1)
    foldindex = 0
    n=len(alldata)
    alldata_array=np.array(alldata)
    foldederrorrate=np.array([])
    X=alldata_array[:,0:featuresno]
    y=alldata_array[:,featuresno:featuresno+labelno]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    lrlist=[0.05]#,0.09,0.1,0.2,0.3,0.4,0.5]
    scalelist=[2,10,30,50]
    hnlist=[featuresno+100]
    bestvector=[0,0,0,0,0]
    avresult=0
    bestvresult=0
    for hn1 in hnlist: 
        for lr1 in lrlist:
            for scl in scalelist:
                avresult,bestnet=CrossValidationAvg(kfold,foldindex,n,foldederrorrate,X_train,y_train,featuresno, hn1, labelno,labelvalue,lr1,scl,epochs,bestvector,Datasetfilename,trainingdataresult,recurrent)
                print('crossv Prediction=%f , lr=%f',(avresult,lr1))
                if(avresult>bestvresult):
                    bestvresult=avresult
                    bestvector=[bestnet,lr1,hn1,bestvresult,scl]
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    # with open(Datasetfilename+str(timestamp)+'.txt', 'a') as f:
    print(">>>>>>>>>>>>>>>>>>>>>>>>"+Datasetfilename+"<<<<<<"+str(timestamp)+"<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>Best Parameters<<<<<<<<<")
    print(">>>>>>>>Best Vector Data<<<<<<<<<")
    print('scale=%f,best result=%f',(bestvector[4],bestvresult))
    print(">>>>>>>>Testing data result<<<<<<<<<")
    X_test_norm = zscore(X_test, axis=0)
    iterationoutput=predict(X_test_norm, y_test, bestvector[0], labelno,labelvalue,bestvector[4],hn1,recurrent)
    print('Final Prediction=%f',iterationoutput)
    print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return avresult


def predict(test_fold_features,test_fold_labels, net, n_outputs,labelvalue,bx,hn,recurent):
    iterationoutput=np.array([])
    statelayer=list()
    for i, row in enumerate((test_fold_features)):
        xxx1=list(row)       
        testfoldlabels=list(test_fold_labels[i])
        predicted,statelayer= forward_propagation(net, xxx1,testfoldlabels, n_outputs,labelvalue,bx,hn,statelayer,recurent)
        iterationoutput=np.append(iterationoutput,[calculateoutputTau([predicted,np.array(testfoldlabels)])])      
    avrre=sum(iterationoutput)/len(test_fold_features) 
    return  avrre





def loadData( featuresno, labelno,labelvalue, iteration,lrate,hn,scale,recurrent):



    train_images = mnist.train_images()[:500]
    train_labels = mnist.train_labels()[:500]
    test_images = mnist.test_images()[:100]
    test_labels = mnist.test_labels()[:100]

    # conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
    # pool = MaxPool2()                  # 26x26x8 -> 13x13x8
    # softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10
    flatterd=[]
    for tup in train_images:
       flatterd.append(tup.ravel() )

    # dataarray = np.asarray(data)
    # allda  taarray = np.asarray(alldata)
    train_labels = train_labels.tolist() # [map(float, i) for i in y]
    train_features = flatterd #[map(float, i) for i in X]

    tot_error=trainingNoValidation(iteration, labelno, train_features,train_labels, featuresno, labelno,labelvalue,lrate,hn,scale,recurrent)
    # tot_error2=training(filename, iteration, alldata_list, featuresno, labelno,labelvalue,lrate,hn,scale,tot_error,recurrent)
                                      
    print('Done')
    return tot_error
   

###############################################################################################################################
###############################################################################################################################
##################################################################################################################################
data = []
labels = []
alldata = []

# Reading the image  
img = skimage.data.chelsea()  
# Converting the image into gray.  
img = skimage.color.rgb2gray(img)
IMAGES_PATH = 'C:\\Github\\PNN\\Data\\Images'
l1_filter = np.zeros((2,3,3))


l1_filter[0, :, :] = np.array([[[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]]])  
l1_filter[1, :, :] = np.array([[[1,   1,  1],[0,   0,  0],[-1, -1, -1]]]) 



#Best Parameter  hn=50 ,scale=20  lrate 0.01
# train_error = loadData(featuresno=784,labelno=1,labelvalue=10,iteration=500,lrate=0.01,hn=50,scale=20,recurrent=false) 

