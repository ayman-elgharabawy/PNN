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
data = []
labels = []
alldata = []



#Error Function#
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

#Drawing image per each iteration for the video#
def DrawGraph(net1,inputs,expected,hn,preverror,epoch,totalerror,totalepoch,epochs,imageindex):
    fig = plt.figure(figsize=cm2inch(19, 24))
    fig.suptitle('SGPN Ranking', fontsize=14, fontweight='bold')
    errline=[0] * 500
    for u in range(len(preverror)):
       errline[u]=preverror[u]
    ############################################################
    axes2 = fig.add_axes([0.1, 0.5, 0.3, 0.4]) # inset axes
    axes2.plot(list(range(0, len(preverror))), preverror, 'g')
    axes2.set_xlabel('No. iterations')
    axes2.set_ylabel('Spearman Correlation')
    axes2.set_ylim([-0.2,1.1])
    axes2.set_title('Avg. Ranking')
    ############################################################
    ax = fig.add_subplot(111)
    #####################################
    input_neurons=len(inputs)
    hidden_neurons=hn
    hidden_layer=net1[0]
    output_layer=net1[1]
    G=nx.Graph()
    for ii in range(input_neurons):
        G.add_node(ii,label=inputs[ii],pos=(0.2,1.6+(ii/1)))
    hh=[]
    for j in range(hidden_neurons):
        G.add_node(j+input_neurons,label=truncate(hidden_layer[j]['result'],3),  pos=(0.8,1.2+(j/2)))
        for k in range(len(hidden_layer[j]['weights'])):
            hh.append(truncate(hidden_layer[j]['weights'][k],3))
    results=[]
    p=0
    ind=1
    for k in range (2):
        p+=k
        subresults=[]
        for j in range(len(expected[k])):
            ind+=1
            p+=j
            res=output_layer[k][j]['result']
            subresults.append(truncate(res,3))
            G.add_node(p+input_neurons+hidden_neurons,label=truncate(res,3),pos=(1.4,k+(ind/2)))
            for w in range(len(output_layer[k][j]['weights'])):
                hh.append(truncate(output_layer[k][j]['weights'][w],3))
        results.append(subresults)        

    node_labels = nx.get_node_attributes(G, 'label')
    pos=nx.get_node_attributes(G,'pos') 
    counter=0
    for i in range(input_neurons):
        for j in range(hidden_neurons):
           G.add_edge(i,j+input_neurons,label=hh[counter],fontsize=8)         
           counter+=1
    edge_labels = nx.get_edge_attributes(G, 'label')       
    for i in range(hidden_neurons):
        p=0
        for k in range (2):
            p+=k
            for j in range(len(expected[k])):
                p+=j
                G.add_edge(i+input_neurons,p+input_neurons+hidden_neurons,label=hh[counter],fontsize=0.05) 
                counter+=1
    edge_labels = nx.get_edge_attributes(G, 'label') 
    weights = [G[u][v]['label']+1 for u,v in edge_labels]
    nx.draw(G, pos=pos,labels=node_labels, with_labels= True,edge_labels=edge_labels,edge_color='#d3d3d3',width=weights, node_size=2000,node_color = '#e5e5e5')
    nx.draw_networkx_edge_labels(G, pos,labels=node_labels, with_labels= True, edge_labels=edge_labels,font_size=6,label_pos=0.25 ,edge_color='#d3d3d3')
    ax.text(0.7, 6.3, 'L.Rate=0.05', fontsize=12)
    ax.text(0.7, 6.1, 'MAFN=10', fontsize=12)
    ax.text(1, 6.1, 'epoch='+str(epoch), fontsize=17)

    ax.text(0.15, 0.9,'Dataset', fontsize=12)
    if(imageindex%2==0):
      ax.text(0.01, 0.7,'[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12) 
      ax.text(0.01, 0.5,'[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12,color='red')
      ax.text(0.65, 0.5,"-->"+str(results[0])+","+str(results[1]), fontsize=12,color='red')
    else:
      ax.text(0.01, 0.7,'[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12,color='red')
      ax.text(0.65, 0.7,"-->"+str(results[0])+","+str(results[1]), fontsize=12,color='red')
      ax.text(0.01, 0.5,'[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12)
  
    for h in range (2):
       t1,t2=ss.spearmanr(results[h],expected[h])
       ax.text(1.5,2.5+(2*h),'Roh'+str(h), fontsize=14)
       ax.text(1.5,2.2+(2*h),str(truncate(t1,2)), fontsize=14)
    ax.text(1.4,3.3,'Avg.Roh='+str(preverror[-1]), fontsize=12)
    # plt.show()
    plt.savefig('C:\\Ayman\\PhDThesis\\video\\test\\' +str(imageindex)+'.png', dpi = 150)
    plt.close(fig)
    plt.close('all') 
    return G

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def calculateoutputTau(iterationoutput):
        sum_Tau=0
        
        for  ii in iterationoutput:
            # tau,pv = ss.spearmanr(ii[1], ii[0]) 
            tau , tauslist = calcAvrTau(ii[0], ii[1],2)  
            # if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

def truncate(n, decimals=0):
    v=[]
    if type(n) is list :
        for i in range(len(n)):
           v.append(float('%.3f'%n[i]))
    else:
        v=float('%.3f'%n)
    return v

def createDropNet(net):
      keep_prob=0.5
      layerdrop=[]    
      for lindex,layer in enumerate(net):
        NeuronDropcache=[]
        if (lindex==0):
            for indexn,neuron in enumerate(layer):
                xx=neuron['weights']
                aa=list(xx)
                NeuronDropcache = list(itertools.chain(NeuronDropcache,aa)) 
            layerdrop.append(NeuronDropcache)
        else:
            totww=[]
            for indnsub,sub1 in enumerate(layer):
                gweight=[]
                for indexn,neuron in enumerate(sub1):
                    xx=neuron['weights']
                    aa=list(xx)
                    gweight.append(aa)
                totww.append(gweight)
            layerdrop.append(totww)      
      layerdrop1=[]
      for lindex1,ldrop in enumerate(layerdrop):
            if (lindex1==0):
                narr=np.array(ldrop)
                NeuronDropcache=[]
                D1 = np.random.uniform(low=-0.9, high=0.9,size=narr.size)
                D1 = D1 < keep_prob
                layerdrop1.append(D1.tolist())  
            else:
                ddlist=[]
                for k in range(len(ldrop)):
                   narr=np.array(ldrop[k])
                   D1 = np.random.uniform(low=-0.9, high=0.9,size=narr.size)
                   D1 = D1 < keep_prob
                   ddlist.append(D1.tolist())
                layerdrop1.append(ddlist)  
      return layerdrop1 #,dropnetperneuron

def print_network(net,epoch,tau,row1,expected):
    with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
        print("------------------------------------------------------ Epoch "+str(epoch)+" ---------------------------------------\n", file=f)
        print("Input row:" +str(row1)+" Expected:"+str(expected), file=f)  
        for i, layer in enumerate(net, 1):
            if (i==1):
                print("=============== Middle layer =============== \n", file=f)           
            else:
                print("=============== Output layer =============== \n", file=f)      
            for j, neuron in enumerate(layer, 1):
                print("Subgroup {} :".format(j), neuron, file=f)  
        print("==== Roh Correlation = "+str(tau)+"======\n", file=f)   
         

def initialize_network(ins, hiddens,noutputlist):
    input_neurons = ins
    hidden_neurons = hiddens
    n_hidden_layers = 1
    net = list()
    subgroup=list()
    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])
            hidden_layers = [{'delta':[],'result':[],'weights': np.random.uniform(low=-0.5, high=0.5,size=hidden_neurons)} for i in range(hidden_neurons)]
            net.append(hidden_layers)
        else:
            first_layer = [{'delta':[],'result':[],'weights': np.random.uniform(low=-0.5, high=0.5,size=input_neurons)} for i in range(hidden_neurons)]
            net.append(first_layer)
    for sub in range(len(noutputlist)):
       subgroup.append([{'delta':[],'result':[],'weights': np.random.uniform(low=-0.5, high=0.5,size=hidden_neurons)} for i in range(noutputlist[sub])])
    net.append(subgroup)
    return net

def forward_propagation(net, input1 ,groupno,noutputlist,noutputvalues,scale):
    row1=[None] * len(noutputlist)
    cache=[]
    if True:
        cache=createDropNet(net)
    for k in range(len(noutputlist)):
        row1[k] = np.asarray(input1)  
    for index,layer in enumerate(net): 
        prev_input=[]  
        if index==0:#MIddle Layer
            for neuron in layer:
                neuron['result']=[]
                for A in range(len(noutputlist)):  
                    xx=neuron['weights']
                    # if True:
                    #     D1=cache[index][neurindex*nn:neurindex*nn+nn]
                    #     aa = aa * np.array(D1)    # Shutdown neurons
                    #     aa = aa / keep_prob    # Scales remaining values
                    # sum1 = np.array(aa).T.dot(row1[0])
                    sum1 = neuron['weights'].T.dot(row1[0])   
                    result1 = SSS(sum1,noutputvalues[A],scale)
                    neuron['result'].append(result1)                    
                prev_input.append(neuron['result'])
        else:    #OutputLayer Layer 
                outtot=[] 
                for indexsub,subgroup in enumerate(layer):
                    prev_input=[]      
                    for subind,neuron in enumerate(subgroup):
                        neuron['result']=[]   
                        xx=neuron['weights']
                        nn=len(xx)
                        aa=list(xx)
                        # if True:
                        #     D1=cache[index][indexsub][subind*nn:subind*nn+nn]
                        #     aa = aa * np.array(D1)    # Shutdown neurons
                        #     aa = aa / keep_prob    # Scales remaining values
                        #     sum1 = aa.T.dot([xx[indexsub] for xx in row1])
                        sum1 = neuron['weights'].T.dot([xx[indexsub] for xx in row1])
                        result1 = SSS(sum1,noutputvalues[indexsub],scale)
                        neuron['result'].append(result1) 
                        prev_input.append(neuron['result'])
                    prev_input =[j for sub in prev_input for j in sub]    
                    outtot.append(prev_input)            
        row1=prev_input

    return outtot ,cache

def back_propagation(net, row, expected,groupno,noutputlist,noutputvalues,scale,cache,dropout):
    for i in reversed(range(len(net))):
        layer = net[i]
        results = list()   
        if i == len(net) - 1:  # output neurons
            results=[] 
            errors = list()
            for indexsub,subgroup in enumerate(layer):
                sub_result=[]      
                for neuron in (subgroup):
                    sub_result.append(neuron['result'])
                sub_result =[j for sub in sub_result for j in sub]      
                results.append(sub_result)  
                output=np.array(results[:][indexsub])
                errors.append(DSpearman(output,expected[indexsub]))
            
            for indexsub,subgroup in enumerate(layer):
                for j,neuron in enumerate(subgroup):
                    neuron['delta']=[]    
                    a =errors[indexsub][j] * dSSS(neuron['result'][0],noutputvalues[indexsub],scale)
                    neuron['delta'].append(a)

        else:  
            errors = list()
            for j in range(len(layer)):
                herror = 0
                nextlayer = net[i + 1] 
                suberrors=[] 
                for indexsub,subgroup in enumerate(nextlayer):
                    sub_result=[]            
                    for nindex,neuron in enumerate(subgroup):
                        zzz=cache[1][indexsub][nindex]
                        if(zzz):
                          herror+= (neuron['weights'][j] * neuron['delta'][0])                                
                    suberrors.append(herror)
                errors.append(suberrors)

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta']=[]
                for g in range(groupno):
                  neuron['delta'].append(errors[j][g] * dSSS(neuron['result'][g],noutputvalues[g],scale))   

def updateWeights(net, input1, lratelist,noutputlist,noutputvalues,cache,dropout):              
    inputs=list()
    for i in range(len(net)):    
        inputs=np.asarray(input1).tolist()     
        if i != 0:#output layer
            inputs=list()
            for h in range(len(noutputlist)):
                   temp1=[neuron['result'][h] for neuron in net[i - 1]] 
                   inputs.append(temp1)
            layer = net[i] 
            for indexsub,subgroup in enumerate(layer):
                counter=0
                for nundex,neuron in enumerate(subgroup): 
                    counter+=nundex
                    for j in range(len(inputs[0])):
                        zzz=cache[1][indexsub][counter]
                        if(zzz):  
                          neuron['weights'][j] -= lratelist[indexsub] * neuron['delta'][0] * inputs[indexsub][j]
                    neuron['weights'][-1] -= lratelist[indexsub] * neuron['delta'][0]
        else: #Middle layer
                for nin,neuron in enumerate(net[i]):                  
                    for j in range(len(inputs)):               
                        zzz=cache[0][j]
                        if(zzz): 
                          for h in range(len(noutputlist)):                                        
                             neuron['weights'][j] -= lratelist[h] * neuron['delta'][h] * inputs[j]
                    for h in range(len(noutputlist)):          
                       neuron['weights'][-1] -= lratelist[h] * neuron['delta'][h]
                 
    return neuron


def calcAvrTau (x_in,y_out,groupno):
    sum_Tau=0    
    Tau_list=[]
    #print('#########################')
    for i in range(len(y_out)):
        tau, p_value = ss.spearmanr(x_in[i], y_out[i])  
        if not np.isnan(tau):
            sum_Tau += tau 
            Tau_list.append(tau)
        else:
            Tau_list=[0,0,0]  
            Tau_list.append(0) 
    return sum_Tau/groupno , Tau_list

def trainingNoValidation( epochs, lratelist,noutputlist,noutputvalues, X, y,noofinputes,hn,scale):
    groupno=len(noutputlist)
    nx=len(X)
    errorRate = []
    imageindex=1
    preverror=[0]
    totalerror=[]
    net = initialize_network(noofinputes, hn,noutputlist)
    totg1=[]
    totg2=[]
    for epoch in range(epochs):
        sum_Tau = 0
        Tau_list=[]
        detailedtau=[]
        errg1=[]
        errg2=[]
        for i, row in enumerate(X):
            outputs,cache = forward_propagation(net, row,groupno,noutputlist,noutputvalues,scale)
            back_propagation(net, row, y[i],groupno,noutputlist,noutputvalues,scale,cache,True)
            updateWeights(net, row, lratelist,noutputlist,noutputvalues,cache,True)
            singleTau,Tau_list = calcAvrTau(y[i], outputs,groupno)
            detailedtau.append(Tau_list)
            sum_Tau+=singleTau
            totalerror.append(preverror[epoch-1])
            # print_network(net,epoch,singleTau,row,y[i])
            # DrawGraph(net,row,y[i],hn,preverror,epoch,totalerror,totalepoch,epochs,imageindex) 
            imageindex+=1
        # print('>epoch=%d,Tau=%.4f ' % (epoch, sum_Tau / nx))
        
        # if epoch % 10 == 0:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print('>epoch=%d,Avg.Tau=%.4f ' % (epoch, sum_Tau / nx))
        xxx=[i[0] for i in detailedtau]
        aa=sum(xxx)/len(detailedtau)
        yyy=[i[1] for i in detailedtau]
        bb=sum(yyy)/len(detailedtau)
        print('Tau1='+str(aa)+' Tau2='+str(bb))#+' Tau3='+str(cc))
        errorRate.append(sum_Tau / nx)
        errg1.append(aa)
        errg2.append(bb)
        totg1.append(errg1)
        totg2.append(errg2)    
        preverror.append(truncate(sum_Tau/nx,2))    
    return totg1 ,totg2

def plotErrorRate(errorRate):
    plt.plot(errorRate)
    plt.ylabel('Error Rate')
    plt.show()



def plot2GError(traing1,traing2,valg1,valg2):
    fig, ax = plt.subplots()
    plt.axes()
    plt.title("Train and Validate model for 2 groups")
    plt.plot(list(range(0,len(traing1))), traing1, label='Train g1',color='blue',marker = 'o')
    plt.plot(list(range(0,len(traing2))), traing2, label='Train g2',color='black',marker = '*')
    plt.plot(list(range(0,len(valg1))), valg1, label='10 fold val. g1',color='green',marker = 'o')
    plt.plot(list(range(0,len(valg2))), valg2, label='10 fold val. g2',color='red',marker = '*')
    plt.xlabel("No. Of iterations ")
    plt.legend()
    plt.ylabel('Roh')
    plt.show()

def plotTrainValidate(trainerror,validateerror):
    fig, ax = plt.subplots()
    plt.axes()
    plt.title("Train-validate")
    plt.plot(list(range(0,len(trainerror))), trainerror, label='Train Error',marker = 'o')
    plt.plot(list(range(0,len(validateerror))), validateerror, label='Validate Error',marker = 'o')
    plt.xlabel("No. Of iterations ")
    plt.legend()
    plt.ylabel('Tau')
    plt.show()

def PNNFit(net,train_fold_features,train_fold_labels,noutputlist,noutputvalues,lratelist,epoch,datalength,hn,b):
    iterationoutput=list()
    sum_Tau=0
    for i, row in enumerate((train_fold_features)):
        xxx1=list(row)       
        trainfoldlbelarray=np.array((train_fold_labels))
        trainfoldexpected=trainfoldlbelarray[i]
        outputs,cache= forward_propagation(net, xxx1,len(noutputlist),noutputlist,noutputvalues,b)
        back_propagation(net, xxx1, trainfoldexpected, len(noutputlist),noutputlist,noutputvalues,b,cache,True)
        updateWeights(net, xxx1, lratelist,noutputlist,noutputvalues,cache,True)   
        iterationoutput.append([trainfoldexpected.tolist(),outputs])

    return iterationoutput , net

def CrossValidationAvg(kfold,foldindex,n,foldederrorrate,X_train,y_train,featuresno, noofhidden, noutputlist,noutputvalues,groupno,lratelist,bbs,epochs,Datasetfilename):
    net = initialize_network(featuresno, noofhidden, noutputlist)
    epocherror=[]
    for idx_train, idx_test in kfold.split(X_train):      
        foldindex += 1
        train_fold_features=X_train[idx_train,:]
        xx=idx_train.tolist()
        yy=np.array(y_train)
        train_fold_labels=yy[xx]
        # train_fold_labels=yy[[xx],:,:].tolist()
        test_fold_features=X_train[idx_test,:]
        xx2=idx_test.tolist()
        test_fold_labels=yy[xx2]
        # test_fold_labels=yy[[xx2],:,:].tolist()
        tot_epoch=[]      
        errorRate_validate=[]
        for epoch in range(epochs):
            iterationoutput_train,net=PNNFit(net,train_fold_features,train_fold_labels,noutputlist,noutputvalues,lratelist,epoch,n,noofhidden,bbs)
            sum_Tau_train = calculateoutputTau(iterationoutput_train)
            tot_epoch.append(epoch)
            epochError_train=sum_Tau_train/(len(train_fold_features))
            errorRate_validate.append(epochError_train)
            if epoch % 10 == 0:
                print('Epoch result >Epoch=%4d ,Tau=%.4f,' % (epoch,epochError_train))
        foldederrorrate=np.append(foldederrorrate,[sum(errorRate_validate)/len(errorRate_validate)]) 
        iterationoutput_test=predict(test_fold_features, test_fold_labels, net, noutputlist,noutputvalues,bbs,groupno)
        result1=calculateoutputTau(iterationoutput_test)
        print('Epoch result >Fold=%4d ,Tau=%.4f,' % (foldindex,result1/(len(test_fold_features))))
        epocherror.append(result1/(len(test_fold_features)))

    epocherrorAvg=  sum(epocherror)/10 
    print('Folded average result > ,Tau=%.4f,' , (epocherrorAvg))
    return epocherrorAvg , net


def training(filename,epochs, X,y, featuresno, noutputlist,groupno,lratelist,noutputvalues,hn,scale):


    kfold = KFold(10, True, 1)
    foldindex = 0
    n=len(alldata)
    alldata_array=np.array(alldata)
    foldederrorrate=np.array([])
    # X=X[:,0:featuresno]
    # y=alldata_array[:,featuresno:featuresno+groupno]
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
                avresult,bestnet=CrossValidationAvg(kfold,foldindex,n,foldederrorrate,X_train,y_train,featuresno, hn, noutputlist,noutputvalues,groupno,lratelist,scale,epochs,filename)   
                print('Final Prediction=%f , lr=%f',(avresult,lr1))
                if(avresult>bestvresult):
                    bestvresult=avresult
                    bestvector=[bestnet,lr1,hn1,bestvresult,scl]
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    # with open(Datasetfilename+str(timestamp)+'.txt', 'a') as f:
    print(">>>>>>>>Best Parameters<<<<<<<<<")
    print(">>>>>>>>Best Vector Data<<<<<<<<<")
    print('scale=%f,best result=%f',(bestvector[4],bestvresult))
    print(">>>>>>>>Testing data result<<<<<<<<<")
    X_test_norm = zscore(X_test, axis=0)
    iterationoutput=predict(X_test_norm, y_test, bestvector[0], noutputlist,noutputvalues,bestvector[4],groupno)
    print('Final Prediction=%f',iterationoutput)
    print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return avresult


def predict(test_fold_features,test_fold_labels, net, noutputlist,noutputvalues,bx,groupno):
    iterationoutput=[]
    testlist=[]
    for i, row in enumerate((test_fold_features)):
        xxx1=list(row) 
        labelsarr=np.array(test_fold_labels)     
        testfoldlabels=labelsarr[i].tolist()
        predicted,cache= forward_propagation(net, xxx1,len(noutputlist),noutputlist,noutputvalues,bx)
        singleTau,Tau_list = calcAvrTau(test_fold_labels[i].tolist(), predicted,groupno)
        testlist.append(Tau_list)


    xxx=[i[0] for i in testlist]
    aa=sum(xxx)/len(testlist)
    yyy=[i[1] for i in testlist]
    bb=sum(yyy)/len(testlist)
    print("============================================")
    print('Prediction==>>Tau1='+str(aa)+' Tau2='+str(bb))#+' Tau3='+str(cc))
    print("============================================")

    return  iterationoutput


def predict_testing(test_fold_features,test_fold_labels, net, noutputlist,bx):
    iterationoutput=[]
    for i, row in enumerate((test_fold_features)):
        xxx1=list(row)  
        labelsarr=np.array(test_fold_labels)     
        testfoldlabels=(labelsarr[i]).tolist()
        predicted,cache= forward_propagation(net, xxx1,len(noutputlist),noutputlist,noutputvalues,bx)
        iterationoutput.append([testfoldlabels,predicted])
    return  iterationoutput    

def rescale(values,featuresno,data_no, new_min , new_max ):
    totaloutput=[] 
    for i in range(featuresno):
        colvalues=[row[i] for row in values]
        old_min, old_max = min(colvalues), max(colvalues)
        outputf = []
        for v in colvalues:
            new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
            outputf.append(new_v)
        totaloutput.append(outputf)
        ############################
    totaloutput1=transpose(totaloutput)    
    return totaloutput1


def loadData(filename, featuresno, noutputlist, epochs,lratelist,noutputvalues,hn,scale):
    gpsTrack = open(filename, "r")
    csvReader = csv.reader(gpsTrack)
    groupno=len(noutputlist)
    data = list()
    labels=[]
    next(csvReader)
    labelno1=noutputlist[0]
    labelno2=noutputlist[1]
    labelno3=noutputlist[2]
    for row in csvReader:
        data.append(list(map(float,row[0:featuresno])))
        a=list(map(float,row[featuresno:featuresno + labelno1]))
        b=list(map(float,row[featuresno+labelno1:featuresno + labelno1+labelno2]))
        c=list(map(float,row[featuresno+labelno1+labelno2:featuresno + labelno1+labelno2+labelno3]))
        ######################################
        ######################################
        labels.append([a,b,c])#,c,d])
    data_no=len(labels)
    train_features_list_norm = rescale(data,featuresno,data_no,-scale,scale) 
    #train_features_list_norm = zscore(data, axis=0)   
    noofhidden = hn
    noofinputes = featuresno
    
    trainederror=[]
###################################

    
# ########################################
    #trainederror1,trainederror2=trainingNoValidation(epochs=epochs, lratelist=lratelist,noutputlist= noutputlist,noutputvalues=noutputvalues,X= train_features_list_norm, y=labels,noofinputes=noofinputes,hn=hn,scale=scale)
    errorpredicted=training(filename=filename, epochs=epochs, X=train_features_list_norm,y=labels,featuresno= featuresno,noutputlist= noutputlist,groupno=groupno,lratelist=lratelist,noutputvalues=noutputvalues,hn=hn,scale=scale)
    
    return trainederror1,trainederror2 ,errorpredicted
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################
# noutputlist=[5,5]
# trainederror,errorpredicted=loadData('RC_Final_5_5',18,noutputlist,1000,0.05,200,1)

# noutputlist=[4,3]
# trainederror,errorpredicted=loadData('germn2005_2009_modified',3,noutputlist,500,0.05,10,1)

# noutputlist=[5,5]
# lratelist=[0.05,0.05]
# # trainederror1,trainederror2 ,errorpredicted=loadData(filename='germn2005_2009_modified',featuresno=31,noutputlist=noutputlist,epochs=1000,lratelist=lratelist,hn=50,scale=20)
# trainederror1,trainederror2 ,errorpredicted=loadData(filename='Data\\SGPNData\\germn2005_2009_modified.csv',featuresno=31,noutputlist=noutputlist,epochs=10,lratelist=lratelist,hn=100,scale=20)

noutputlist=[4,2,2]
noutputvalues=[4,3,2]
lratelist=[0.05,0.05,0.05]
trainederror1,trainederror2 ,errorpredicted=loadData(filename='Data\\SGPNData\\Emotions.csv',featuresno=72,noutputlist=noutputlist,noutputvalues=noutputvalues,epochs=10,lratelist=lratelist,hn=50,scale=20)

print("Done.")

#############################################################################################
# noutputlist=[3,3,5]
# trainederror,errorpredicted=loadData('wine_iris_stock_3_3_5',22,noutputlist,5000,0.05,200,1)
# plotTrainValidate(trainederror,errorpredicted)

# noutputlist=[130,130]
# loadData('RC_Final',18,noutputlist,2000,0.07,100)

# noutputlist=[5,5]
# loadData('germn2005_2009_modified',31,noutputlist,10000,0.07,50,20)

# noutputlist=[3,3,5]
# loadData('wine_iris_stock_3_3_5',22,noutputlist,2000,0.07,100)

# noutputlist=[16,3]
# loadData('wisconsin_iris_16_3',20,noutputlist,2000,0.07,100)


# noutputlist=[3,5]
# loadData('wine_stock_3_5',18,noutputlist,2000,0.07,100)


# noutputlist=[3,5]
# loadData('iris_stock_3_5',9,noutputlist,2000,0.07,100)

# noutputlist=[3,3]
# loadData('wine_iris_3_3',17,noutputlist,20000,0.07,25)
