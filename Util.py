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



def DrawGraph(net1,inputs,expected,hn,epoch,totalerror,totalepoch,epochs):
    fig = plt.figure(figsize=cm2inch(17, 23))
    fig.suptitle('PNN- SS function', fontsize=14, fontweight='bold')
    axes2 = fig.add_axes([0.65, 0.7, 0.3, 0.2]) # inset axes
    axes2.plot(list(range(0,len(totalerror))), totalerror, 'g')
    axes2.set_xlabel('y')
    axes2.set_ylabel('x')
    axes2.set_ylim([-1,1])
    axes2.set_xlim([0,epochs])
    axes2.set_title('Tau Convergence')

    ax = fig.add_subplot(111)
    #####################################
    input_neurons=len(inputs)
    hidden_neurons=hn
    output_neurons=len(expected)
    hidden_layer=net1[0]
    output_layer=net1[1]
    G=nx.Graph()

    for ii in range(input_neurons):
        G.add_node(ii,label=inputs[ii],pos=(0.2,1.6+(ii/1)))

    hh=[]
    for j in range(hidden_neurons):
        G.add_node(j+input_neurons,label=truncate(hidden_layer[j]['result'],2),  pos=(0.8,1.2+(j/2)))
        for k in range(len(hidden_layer[j]['weights'])):
            hh.append(truncate(hidden_layer[j]['weights'][k],2))
    results=[]
    for j in range(output_neurons):
        results.append(truncate(output_layer[j]['result'],2))
        G.add_node(j+input_neurons+hidden_neurons,label=truncate(output_layer[j]['result'],2),pos=(1.4,1.6+(j/2)))
        for k in range(len(output_layer[j]['weights'])):
            hh.append(truncate(output_layer[j]['weights'][k],2))

    node_labels = nx.get_node_attributes(G, 'label')
    pos=nx.get_node_attributes(G,'pos')
    counter=0
    for i in range(input_neurons):
        for j in range(hidden_neurons):
           G.add_edge(i,j+input_neurons,label=hh[counter],fontsize=8)         
           counter+=1
    edge_labels = nx.get_edge_attributes(G, 'label')       

    for i in range(hidden_neurons):
        for j in range(output_neurons):
           G.add_edge(i+input_neurons,j+input_neurons+hidden_neurons,label=hh[counter],fontsize=8) 
           counter+=1
    edge_labels = nx.get_edge_attributes(G, 'label') 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=6, label_pos=0.75,edge_color='#d3d3d3')
    nx.draw(G, pos=pos,labels=node_labels,edge_labels=edge_labels,edge_color='#d3d3d3', with_labels= True, node_size=2000,node_color = '#e5e5e5')
    
    ax.text(1.2, 0.5, 'epoch='+str(epoch), fontsize=17)
    ax.text(1.2, 3, 'Output  Output Expected', fontsize=12)
    ax.text(1.4, 2.9, 'Rank   Rank',   fontsize=12)

    ax.text(1.65 ,2.6,expected[2], fontsize=12)
    ax.text(1.65, 2.1,expected[1], fontsize=12)
    ax.text(1.65, 1.6,expected[0], fontsize=12)
    rankedd=rankdata(results)
    ax.text(1.5 ,2.6,rankedd[2], fontsize=12)
    ax.text(1.5, 2.1,rankedd[1], fontsize=12)
    ax.text(1.5, 1.6,rankedd[0], fontsize=12)

    x = np.linspace(1.6,1.6)
    y = np.linspace(1.5,2.7)

    ax.text(0.15, 1.1,'Dataset', fontsize=12)
    ax.text(0.15, 1,'[0.1,0.1,0.1] [1,2,3]', fontsize=12)
    ax.text(0.15, 0.9,'[0.5,0.5,0.5] [3,2,1]', fontsize=12)
    # ax.text(0.15, 0.8,'[0.166,0.0,0.186] [1,2,3]', fontsize=12)  
    plt.plot(x, y)
    t1,t2=ss.kendalltau(results,expected)
    ax.text(1.2,1.1,'Current Tau='+str(truncate(t1,2)), fontsize=14)
    ax.text(0.2,0.8,'Avg. Tau='+str(removeDuplicates(totalerror)), fontsize=12)
    ax.text(0.2,0.7,'Epochs='+str(removeDuplicates(totalepoch)), fontsize=12)
    plt.show()
    plt.savefig('C:\\Ayman\\PhDThesis\\test\\' +str(epoch)+'.png')
    return G

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def removeDuplicates(listofElements):
    
    # Create an empty list to store unique elements
    uniqueList = []
    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)
    # Return the list of unique elements        
    return uniqueList


    def truncate(n, decimals=0):
    if  np.isnan(n):
        return 0
    multiplier = 100 ** decimals
    v=int(n * multiplier) / multiplier
    return v

# def PlotRMSRankError(tauerror1,tauerror2,rms1,rms2,errtype1,errtype2,dropout,Afunction,Description):
#     fig, ax = plt.subplots()
#     plt.axes()
#     plt.plot(list(range(0,len(rms1))), rms1, color='blue', label=errtype1+'Type - RMS',marker = 'x') 
#     plt.plot(list(range(0,len(tauerror1))), tauerror1, color='blue', label=errtype1+' Tau',marker = 'o')

#     plt.plot(list(range(0,len(rms2))), rms2, color='green', label=errtype2+'Type - RMS',marker = 'x') 
#     plt.plot(list(range(0,len(tauerror2))), tauerror2, color='green', label=errtype2+' Tau',marker = 'o')
#     plt.xlabel("No. Of iterations ")
#     plt.title("PNN Type A Objective function types performance ,Dropout= "+str(dropout)+' '+Description)
#     plt.legend()
#     plt.show()

# def PlotRMSRankError(tauerror1,rms1,errtype1,dropout,Afunction,Description):
#     fig, ax = plt.subplots(figsize=cm2inch(30, 23))
#     plt.axes()
#     plt.plot(list(range(0,len(rms1))), rms1, color='blue', label=errtype1+'Type - RMS',marker = 'x') 
#     plt.plot(list(range(0,len(tauerror1))), tauerror1, color='blue', label=errtype1+' Tau',marker = 'o')

#     plt.xlabel("No. Of iterations ")
#     plt.title("PNN Type A Objective function types performance ,Dropout= "+str(dropout)+Description)
#     plt.legend()
#     plt.savefig('C:\\Ayman\\PhDThesis\\method\\' +Description+'.png')
#     # plt.show()

def PlotErrorNoValidation2errors(trainingdataresult1,trainingdataresult2):
    fig, ax = plt.subplots()
    plt.axes()
    plt.plot(list(range(0,len(trainingdataresult1))), trainingdataresult1,color='black', label='Tau - model- Without dropout',marker = 'x')
    plt.plot(list(range(0,len(trainingdataresult2))), trainingdataresult2,color='blue', label='Tau - model with dropout',marker = 'o')
    plt.xlabel("No. Of iterations ")
    plt.legend()
    plt.ylabel('Tau')
    plt.show()

def PlotError(trainingdataresult,tauerror1,dropout,Afunction):
    fig, ax = plt.subplots()
    plt.axes()
    plt.title(" using Dropout= "+str(dropout))
    # for foldindex,errorfold in enumerate(tauerror1):
    #   plt.plot(list(range(0,len(errorfold))), errorfold, label='Tau - Fold'+str(foldindex),marker = 'o')
    plt.plot(list(range(0,len(tauerror1))), tauerror1,color='blue', label='Tau - predict model',marker = 'x')
    plt.plot(list(range(0,len(trainingdataresult))), trainingdataresult,color='black', label='Tau - Training model',marker = 'x')
    plt.xlabel("No. Of iterations ")
    plt.legend()
    plt.ylabel(r'$\rho$')
    plt.show()

def plotData(iterationoutput,tau,dropout,Afunction,epoch,lrate,hn):

    output1=([i[0] for i in iterationoutput])
    output2=([i[1] for i in iterationoutput])
    output3=([i[2] for i in iterationoutput])
    # output4=([i[3] for i in iterationoutput])
    fig, ax = plt.subplots()
    scatter = ax.scatter(output1, output3, c=output2, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Label Ranking")
    ax.add_artist(legend1)
    legend2 = ax.legend( loc="lower right", title="\n"+tau+"\n"+epoch+"\n"+lrate+"\n"+hn)
    ax.add_artist(legend2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_coords(0.41,1)
    plt.ylim(-0.1,6)
    plt.xlim(-10,10)
    plt.xlabel(u"Sum(weights X Inputs) of middle layer")
    plt.ylabel("SS output")                       # title
    plt.show()