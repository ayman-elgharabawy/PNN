import numpy as np
from sklearn import preprocessing 
import math
# from IPython.display import Image,display
import matplotlib.pyplot as plt
import scipy.stats
# import pandas as pd
from itertools import combinations, permutations
import csv
import scipy.stats as ss
from sympy import *
import pandas as pd 
import numpy as np
from itertools import *

from sklearn.decomposition import PCA
data = []
labels = []
alldata = []


def numericlabels(data):
    numericlabels = list()
    numericrow = list()
    for i in range(0, (len(data))):
        row = data[i]
        temprow = list()
        if (len(row) == 0):
            del data[i]
            continue
        for j in range(0, (len(row))):
            temp = row[j]
            temprow.append(temp[1:])
            temprownumeric = [int(item) for item in temprow]
        numericrow.append(temprownumeric)
    return numericrow


def print_network(net):
    for i, layer in enumerate(net, 1):
        print("Layer {} ".format(i))
        for j, neuron in enumerate(layer, 1):
            print("neuron {} :".format(j), neuron)


def initialize_network(ins, hiddens, outs, n_hlayers):
    input_neurons = ins
    hidden_neurons = hiddens
    output_neurons = outs
    n_hidden_layers = n_hlayers
    net = list()
    for h in range(n_hidden_layers):
        if h != 0:
            input_neurons = len(net[-1])

        hidden_layer = [{'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons)]
        net.append(hidden_layer)

    output_layer = [{'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)

    return net


def getOutputWeights(outout):
    weights = list()
    for i in range(0, len(outout)):
        acc = 1
        for j in range(0, len(outout)):
            if (i != j):
                if (outout[i] > outout[j]):
                    acc += 1
        weights.append(acc)  # *0.001)
    return weights



def extractWeights(output):
    arr = []
    for i in range(len(output)):
        arr.append(output[i]['weights'])
    return arr


def extractPairs(y_labels):
    pairs = []
    for i in range(0, len(y_labels)):
        for j in range(i + 1, len(y_labels)):
            if (i != j):
                if (y_labels[i] > y_labels[j]):
                    pairs.append([i, j])
                else:
                    pairs.append([j, i])
    return pairs


def kendalltau_derivative_index(index,layer):
   
    z=0
    for i in range (len(layer)):
        dev=0
        if(index==i):
            for j in range (len(layer)):
                if(i!=j):
                    if(i<j):
                        z+=(-math.pow(1-np.tanh((layer[i]['result']-layer[j]['result'])),2))-np.tanh(100*(expected[i]-expected[j]))
                    else:
                        z+=(math.pow(1-np.tanh((layer[j]['result']-layer[i]['result'])),2))-np.tanh(100*(expected[j]-expected[i]))
                    dev+=z
    return dev


# def getTotalError(output,expected):  
#     tanhrow=list()
#     n=len(expected)
#     if(len(output)==len(expected)):  
#         for i in range (len(output)):
#             sum1=0
#             for j in range (len(output)):
#                 if i!=j:
#                     if(i<j):
#                       sum1+=np.tanh(100*(output[i]-output[j]))-np.tanh(100*(expected[i]-expected[j]))
#                     else:
#                       sum1+=np.tanh(100*(output[j]-output[i]))-np.tanh(100*(expected[j]-expected[i]))
#             tanhrow.append(sum1)      
#     else:
#         print("whats up wrong")

#     return tanhrow

# def stairs_derivative(xi, n_outputs):

#     n = n_outputs
#     derivative = 0
#     #derivative=0.1*(49* np.power(sech(8 - (10*xi)),2) + 49*np.power(sech(6+ (10 *xi)),2))
#     derivative=0.35*(10.792 *np.power(sech(81.6 + 7.6 *(-11.4 + xi)),2) + 10.792  * np.power(sech(90.8  + 7.6 *(-11.4 + xi)),2))
#     return derivative
def stairs_derivative7(xi, n_outputs):

    derivative = 0.63*(0.8*(1-np.power(np.tanh(31.6*xi + (47 - 76/6)),2)) + 
    0.8*(1-np.power(np.tanh(31.6*xi + (47 - 2*76/6)),2)) + 
    0.8*(1-np.power(np.tanh(31.6*xi + (47 - 3*76/6)),2)) + 
    0.8*(1-np.power(np.tanh(31.6*xi + (47 - 4*76/6)),2)) + 
    0.8*(1-np.power(np.tanh(31.6*xi + (47 - 5*76/6)),2)) + 
    0.8*(1-np.power(np.tanh(31.6*xi + (47 - 6*76/6)),2)) + 6/2 + 3.30)

    return derivative

def stairs_derivative10(xi,n_outputs):

    n = n_outputs
    derivative = 0
    derivative =0.63*(0.8*(1-np.power(np.tanh(35*xi + (45 - 76/9)),2)) + 0.8*(1-np.power(np.tanh(35*xi + (45 - 2*76/9)),2)) + 
    0.8*(1-np.power(np.tanh(35*xi + (45 - 3*76/9)),2)) + 0.8*(1-np.power(np.tanh(35*xi + (45 - 4*76/9)),2)) + 
    0.8*(1-np.power(np.tanh(35*xi + (45 - 5*76/9)),2)) + 0.8*(1-np.power(np.tanh(35*xi + (45 - 6*76/9)),2)) + 
    0.8*(1-np.power(np.tanh(35*xi + (45 - 7*76/9)),2)) + 0.8*(1-np.power(np.tanh(35*xi + (45 - 8*76/9)),2)) + 
    0.8*(1-np.power(np.tanh(35*xi + (45 - 9*76/9)),2)) + 9/2 + 4.2)

    return derivative

def back_propagation(net, row, expected, outputs, n_outputs):
    for i in reversed(range(len(net))):
        layer = net[i]
        results = list()
        errors = np.array([])

        if i == len(net) - 1:  # output neurons
            results = [neuron['result'] for neuron in layer]
            errors = ((expected) - np.array(results))/100 # getOutputWeights(results))
            #errors=getTotalError(results,expected)

        else:
            for j in range(len(layer)):
                herror = 0
                nextlayer = net[i + 1]
                for neuron in nextlayer:
                    herror += (neuron['weights'][j] * neuron['delta'])
                errors = np.append(errors, [herror])

        for j in range(len(layer)):
            neuron = layer[j]
            results = [neuron1['result'] for neuron1 in layer]
            neuron['delta'] =errors[j] * stairs_derivative10(neuron['result'], n_outputs)# * kendalltau_derivative_index(j,layer)


def updateWeights(net, input, lrate):
    for i in range(len(net)):
        inputs = input
        if i != 0:
            inputs = [neuron['result'] for neuron in net[i - 1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += lrate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += lrate * neuron['delta']

    return neuron


def countMisorderedPairs(outputpairs, expectedpairs):
    changes = [item for item in outputpairs if item not in expectedpairs]
    return len(changes)


def training(net, epochs, lrate, n_outputs, X, y):
    errorRate = []
    errors = []

    for epoch in range(epochs):
        sum_Tau = 0
        sum_conflict = 0
        for i, row in enumerate(X):
            outputs = forward_propagation(net, row, y[i], n_outputs)
            expectedpairs = extractPairs(y[i][:])
            outputpairs = extractPairs(outputs)
            back_propagation(net, row, y[i], outputs, n_outputs)
            updateWeights(net, row, lrate)
            tau, p_value = ss.kendalltau(y[i], outputs)
            #print("==tue==outputs==expected",tau,outputs,y[i])
            if not np.isnan(tau):
                sum_Tau += tau
            sum_conflict += countMisorderedPairs(outputpairs, expectedpairs)
            # print(outputs)
        if epoch % 1000 == 0:
            print('>epoch=%d,Tau=%.4f, conflict pairs=%.4f' % (epoch, sum_Tau / len(X), sum_conflict))
            errorRate.append(sum_Tau / len(X))
            # print(outputs)
            #print_network(net)
    plotErrorRate(errorRate)


# Make a prediction with a network# Make a
# def predict(network, row,net):
#     outputs = forward_propagation(net, row,n_outputs)
#     return outputs

def plotErrorRate(errorRate):
    plt.plot(errorRate)
    plt.ylabel('Error Rate')
    plt.show()


def forward_propagation(net, input, pred, n_outputs):
    row = input
    for layer in net:
        prev_input = np.array([])
        for neuron in layer:
            sum = neuron['weights'].T.dot(row)
            result = Activation10(sum, n_outputs)
            neuron['result'] = result
            prev_input = np.append(prev_input, [result])
        row = prev_input

    return row


def Activation7(xi, n_outputs):

    n = n_outputs
    sum = 0
    sum =0.63*(0.8*np.tanh(31.6*xi + (47 - 76/6)) + 
    0.8*np.tanh(31.6*xi + (47 - 2*76/6)) + 
    0.8*np.tanh(31.6*xi + (47 - 3*76/6)) + 
    0.8*np.tanh(31.6*xi + (47 - 4*76/6)) + 
    0.8*np.tanh(31.6*xi + (47 - 5*76/6)) + 
    0.8*np.tanh(31.6*xi + (47 - 6*76/6)) + 6/2 + 3.30)

    return sum

def Activation10(xi, n_outputs):
    
    n = n_outputs
    sum = 0
    sum =0.63*(0.8*np.tanh(35*xi + (45 - 76/9)) + 0.8*np.tanh(35*xi + (45 - 2*76/9)) + 
    0.8*np.tanh(35*xi + (45 - 3*76/9)) + 0.8*np.tanh(35*xi + (45 - 4*76/9)) + 
    0.8*np.tanh(35*xi + (45 - 5*76/9)) + 0.8*np.tanh(35*xi + (45 - 6*76/9)) + 
    0.8*np.tanh(35*xi + (45 - 7*76/9)) + 0.8*np.tanh(35*xi + (45 - 8*76/9)) + 
    0.8*np.tanh(35*xi + (45 - 9*76/9)) + 9/2 + 4.2)

    return sum


def loadData(filename, featuresno, labelno, iteration, step):
    filename ='C:\\Research\PhDThesis\\' + filename + '.txt'
    gpsTrack = open(filename, "r")
    csvReader = csv.reader(gpsTrack)
    data = list()
    labels = list()
    alldata = list()

    for row in csvReader:
        data.append(row[0:featuresno])
        labels.append(row[featuresno:featuresno + labelno])
        alldata.append(row[:])

    labelsarray = np.asarray(labels)
    dataarray = np.asarray(data)
    alldataarray = np.asarray(alldata)
    labels = [map(float, i) for i in labelsarray]
    data = [map(float, i) for i in dataarray]
    alldata = [map(float, i) for i in alldataarray]
    #y=[[1,3,2],[3,2,1],[1,3,2]]#,[2,1,4,3],[4,3,2,1],[1,4,3,2],[3,2,4,1]]
    y = np.array(labels)
    #X=[[0.1,0.2,0.3],[0.3,0.1,0.2],[0.1,0.3,0.2]]#,[2,1,4,3],[4,3,2,1],[1,4,3,2],[3,2,4,1]]
    X = np.array(data)  # rankdata(np.array(data))
    nooflabels = len(y[0])
    noofhidden = len(y[0]) + 12
    noofinputes = len(X[0])
    net = initialize_network(noofinputes, noofhidden, nooflabels,1)
    training(net, iteration, step, nooflabels, X, y)


def getkeyindexbefor(rankmap,val):
    llist=rankmap.keys().tolist()
    for i in range(len(llist)):
      if(val in (llist.keys()).index()):
          s=rankmap.keys().index(llist[i]) 
          return s
   

def getrank(stringval):
    rankmap = {
		 "a" : 1,"b" : 2,"c" : 3,"d" : 4,"e" : 5,"f" : 6,"g" : 7,"h" : 8,"i" : 9,"j" :10
        #  ,"k" : 10,"l" : 11,"m" : 12,"n" : 13,"o" : 14,"p" : 15,"q" : 16,"r" : 0,"s" : 0,"t" : 0,"u" : 0,"v" : 0,
        # "w" : 0,"x" : 0,"y" : 0,"z" : 0
	}
    counter=0
    for i in range(len(stringval)):
        if(stringval[i]!='<' and stringval[i]!='>'):
         counter+=1
         xxx=stringval[i]
         rankmap[xxx]=counter


    return list(rankmap.values())


def labelsprocessing(labelsarray):
    ranklist=list()
    for i in  labelsarray:
       rankmap=getrank(i)
       ranklist.append(rankmap)    
    return ranklist


# def groundTruthBySubgroup(alldata,featuresno,labelno):

    # for i in alldata:
    #         writer.writerow(i[0])



def indexingLabels(filename):

    file_handler = open(filename, "r")  
    data = pd.read_csv(file_handler, sep = ",") 
    file_handler.close() 

    alldata= pd.concat([data.sex,data.age,data.answer_duration,data.lived_in_prefecture,data.lived_in_region,data.lived_in,data.lives_in_prefecture,data.lives_in_region,data.lives_in,data.changed_city],1)
    labelsarray= data.ranking        

    labels=labelsprocessing(labelsarray)
    y = np.array(labels).tolist()
    X = np.array(alldata).tolist()

    alldata= [[i+j] for i,j in zip(X,y)]
    tot=[]
    columns= ['sex','age','answer_duration','lived_in_prefecture','lived_in_region','lived_in','lives_in_prefecture','lives_in_region','lives_in','changed_city','a','b','c','d','e','f','g','h','i','j']
    for i in alldata:
           tot.append(i[0])
    df = pd.DataFrame(tot) 
    df.columns=columns
    df.to_csv('Data\\LRData\\sushi_ranked.csv', sep=',', encoding='utf-8')      
    print('Converted')
    return tot
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################
from sklearn.model_selection import train_test_split
import itertools


alldata=indexingLabels('Data\\LRData\\sushi.txt') 
# groundTruthBySubgroup(alldata,10,10)