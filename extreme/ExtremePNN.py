# Use Python 3.7.3
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import zscore
from itertools import combinations, permutations
import csv
import scipy.stats as ss
import random
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import itertools
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime

class ExtremePNN:

    def __init__(self):
        print("PNN Starting..")

    def SSS(self,xi,nlabel,bx):
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

    def dSSS(self,xi,nlabel,bx): 
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

    def Spearman(self,output,expected):
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

    def DSpearman(self,output,expected):
        n=len(expected)
        dif=0
        diflist=np.array([])
        deflist=np.array([])
        for i in range (n):
            o=output[i]
            e=int(expected[i])
            diflist=np.append(diflist,[2*(o-e)])
            dif+=2*(o-e)
        
        den=n*(np.power(n,2)-1)
        for dd in diflist:
            deflist=np.append(deflist,[((dd)/(den))])
        return deflist  

    def binarySplitter(self,labs):
        newlist=[]
        for lab in labs:
            if( lab==1):
                newlist.append([2,1])
            else:
                newlist.append([1,2])   
        return newlist

    def print_network(self,net):#,epoch,tau,row1):
        # with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
            # print("------------------------------------------------------ Epoch "+str(epoch)+" ---------------------------------------\n")
            # print("Input row:" +str(row1))  
            for i, layer in enumerate(net, 1):
                if (i==1):
                    print("=============== Middle layer =============== \n")  
                    for neuron in (net[0]['middle']):
                       print("Weights  :"+str(neuron['weights'])  )  
                       print("delta  :"+str(neuron['delta'] ) )  
                       print("result  :"+str(neuron['result'] ))       
                else:
                    print("=============== Output layer =============== \n")      
                    for neuron in (net[1]['output']):
                       print("Weights  :"+str(neuron['weights'])  )  
                       print("delta  :"+str(neuron['delta'] ) ) 
                       print("result  :"+str(neuron['result'] ))
            # print("==== Roh Correlation = "+str(tau)+"======\n")       
        
    def initialize_networks(self,ins, hiddens, nolabels):
        
        input_neurons = ins
        hidden_neurons = hiddens
        output_neurons = 2
        nets=[]
        for i in range(nolabels):
            net = list()
            hidden_layer = {'middle':[ {'weights': np.random.uniform(low=-0.5, high=0.5,size=input_neurons)} for i in range(hidden_neurons)] }
            net.append(hidden_layer)
            output_layer = {'output':[{'weights': np.random.uniform(low=-0.5, high=0.5,size=hidden_neurons)} for i in range(output_neurons)] }   
            net.append(output_layer)
            nets.append(net)
        return nets 


    def forward_propagation(self,net, input1,trainfold, n_outputs,labelvalue,b,statelayer):
        recuurent=False
        row1 = input1
        prev_input = np.array([])
        for nindex,neuron in enumerate(net[0]['middle']):
            if(recuurent):  #recuurent
               sum1 = neuron['weights'].T.dot(row1)+statelayer[nindex]
            else:
               sum1 = neuron['weights'].T.dot(row1)
               if (np.isnan(sum1)):
                   sum1=0
            result = self.SSS(sum1,labelvalue,b)  
            if np.isnan(result):
                result=0    
            neuron['result'] = result 
            prev_input = np.append(prev_input, [result])    
        row1 = prev_input 
        statelayer=prev_input 
        
        prev_input = np.array([])
        for neuron in net[1]['output']:
            sum1 = neuron['weights'].T.dot(row1)
            result = self.SSS(sum1,labelvalue,b)   
            if np.isnan(result):
                result=0   
            neuron['result'] = result 
            prev_input = np.append(prev_input, [result])
    
        row1 = prev_input 
        return row1 ,statelayer


    def back_propagation(self,net, features_no, expected, outputs, n_outputs,labelvalue,b):

        results = list()
        errors = np.array([]) 
        results = [neuron['result'] for neuron in net[1]['output']]
        errors =self.DSpearman(results,expected)

        for indexsub,neuron in enumerate(net[1]['output']):   
            neuron['delta'] =errors[indexsub] * self.dSSS(neuron['result'],labelvalue,b)
            if(np.isnan(neuron['delta'])):
                    neuron['delta']=0

        for  j ,neuron in enumerate(net[0]['middle']):
            herror = 0            
            for outneuron in (net[1]['output']):            
                herror += (outneuron['weights'][j]) * (outneuron['delta'])    
            errors=np.append(errors,[herror])

        for indexsub,neuron in enumerate(net[0]['middle']):
            neuron['delta'] =errors[indexsub] * self.dSSS(neuron['result'],labelvalue,b)
            if(np.isnan(neuron['delta'])):
                neuron['delta']=0


    def updateWeights(self,net, input1, lrate):

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

        return net

    def PNNFit(self,nets,epochs,train_fold_features,train_fold_labels,features_no,n_outputs,labelvalue,lrate,hn,b):
        iterationoutput=np.array([])
        statelayer=[0]*hn 
        trainfoldexpected= self.binarySplitter(train_fold_labels)
        for epoch in range(epochs):
            for i, row in enumerate((train_fold_features)):
                newnets=[]
                for index,rownet in enumerate(nets):
                    xxx1=np.array(list(row))                          
                    outputs ,statelayer = self.forward_propagation(rownet, xxx1,trainfoldexpected[index], n_outputs,labelvalue,b,statelayer)
                    self.back_propagation(rownet, features_no, trainfoldexpected[index], outputs, n_outputs,labelvalue,b)
                    net=self.updateWeights(rownet, xxx1, lrate) 
                    newnets.append(net)  
            nets=newnets   
            # cc=self.calculateoutputTau([outputs,np.array(trainfoldexpected)])
            # iterationoutput=np.append(iterationoutput,cc)
        final_nets=nets    
        # z=len(train_fold_features)    
        # rr=sum(iterationoutput/z)    
        return final_nets#,rr 

    def calculateoutputTau(self,iterationoutput):
            sum_Tau=0
            tau,pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])   
            if not np.isnan(tau):
                sum_Tau += tau
            return sum_Tau

    def CrossValidationAvg(self,kfold,foldcounter,foldindex,X_train,y_train,featuresno, noofhidden, labelno,labelvalue,lrate,bbs,epochs,bestvector):
        nets = self.initialize_networks(featuresno, noofhidden, labelno)
        avr_res=0
        tot_etau=0
        for idx_train, idx_test in kfold.split(X_train):      
            foldindex += 1
            trainlabel=[]
            testlabel=[] 
            trainfeatures=[]
            testfeaatures=[] 
            trainingnets=[] #
            testingnets=[]    
            for i in idx_train:
                trainlabel.append(y_train[i]) 
                trainfeatures.append(X_train[i])
                trainingnets.append(nets[i])
            for i in idx_test:
                testlabel.append(y_train[i]) 
                testfeaatures.append(X_train[i])
                testingnets.append(nets[i])
            nets=self.PNNFit(trainingnets,epochs,trainfeatures,trainlabel,featuresno,labelno,labelvalue,lrate,noofhidden,bbs)
            # self.print_network(net)
            iterationoutput=self.predict(testingnets,testfeaatures,testlabel,  labelno,labelvalue,bbs,noofhidden)
            # self.print_network(net)
            print("-- Predition one fold Result %d",iterationoutput)
            tot_etau+=iterationoutput
        avr_res=tot_etau/foldcounter  
        print("Final average %.2f Folds test Result %.8f",foldcounter,avr_res)
            
        return avr_res, net
    

    def training(self,epochs, X,y, featuresno, labelno,labelvalue,lrate,hn,scale):
        foldcounter=10
        kfold = sklearn.model_selection.KFold(foldcounter,shuffle= True,random_state= 1)
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
                    avresult,bestnet=self.CrossValidationAvg(kfold,foldcounter,foldindex,X,y,featuresno, hn1, labelno,labelvalue,lr1,scl,epochs,bestvector)
                    print('crossv Prediction=%f , lr=%f',(avresult,lr1))
                    if(avresult>bestvresult):
                        bestvresult=avresult
                        bestvector=[bestnet,lr1,hn1,bestvresult,scl]
        now = datetime.now()
        timestamp = datetime.timestamp(now)


        print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return bestnet,avresult


    def predict(self,nets,test_fold_features,test_fold_labels, n_outputs,labelvalue,bx,hn):
        iterationoutput=np.array([])
        statelayer=list()
        for i, row in enumerate((test_fold_features)):
            for index,rownet in enumerate(nets): 
                xxx1=np.array(list(row))    
                testfoldlabels=list(test_fold_labels[i])
                predicted,statelayer= self.forward_propagation(rownet, row,testfoldlabels, n_outputs,labelvalue,bx,statelayer)
                iterationoutput=np.append(iterationoutput,[self.calculateoutputTau([predicted,np.array(testfoldlabels)])])      
        avrre=sum(iterationoutput)/len(test_fold_features) 
        return  avrre



    def loadData1(self,filename, featuresno, labelno,labelvalue, iteration,lrate,hn):
        data = list()
        labels = list()
        alldata = list()
        scale=2*labelno
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
        net1=self.loadData(X.astype(np.float32),y.astype(np.int32), featuresno, labelno,labelvalue, iteration,lrate,hn,scale)                       
        print('Done')
        return net1

    def loadData(self,X,y, featuresno, labelno,labelvalue, iteration,lrate,hn,scale):
        features_norm = zscore(X, axis=1)
        net1,tot_error2=self.training(iteration, features_norm,y, featuresno, labelno,labelvalue,lrate,hn,scale)                           
        print('Done')
        return net1
 ###############################################################
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################

