# Copyright 2021 Ayman Elgharabawy. All Rights Reserved.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
from scipy.stats.mstats import spearmanr
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime


class PAOneLayer1:

    def __init__(self):
        print("AutoEncoder One Layer Starting..")

    def SSS(self,xi,n,boundaryValue):
        sum = 0
        c=100
        b=100/boundaryValue
        for i in range(n):
            sum +=-0.5*(np.tanh((-b*(xi))-(c*(1-((2*i)/(n-1))))))  
        sum= sum+(n*0.5)  
        return sum     

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


    def PSS(self, xi, n, stepwidth=2):
        sum1 = 0
        b = 100
        for i in range(n):
            sum1 += -0.5 * (np.tanh(-b * (xi - (stepwidth * i))))
        sum1 = sum1 + (n / 2)
        return sum1

    def dPSS(self, xi, n, stepwidth=2):
        sum1 = 0
        b = 100
        for i in range(n):
            sum1 += -0.5 * (1 - np.power(np.tanh(-b * (xi - (stepwidth * i))), 2))
        sum1 = sum1 + (n / 2)
        return sum1

    def Spearman(self, output, expected):
        n = len(expected)
        dif = 0
        diflist = np.array([])
        for i in range(n):
            diflist = np.append(diflist, [np.power(output[i] - expected[i], 2)])
            dif += np.power(output[i] - expected[i], 2)

        den = n * (np.power(n, 2) - 1)
        deflist = np.array([])
        for dd in diflist:
            deflist = np.append(deflist, ((6 * dd) / (den)))
        return deflist

    def DSpearman(self, output, expected):
        nn = len(output[0])
        diflist=[]
        deflist = np.array([])
        diflist = [2 * (output[0][i] - (expected[0][i])) for i in range(nn)]
        diflist=np.array(diflist)
        den = nn * (np.power(nn, 2) - 1)
        for dd in diflist:
            deflist = np.append(deflist, [((dd) / (den))])
        new = np.reshape(diflist, (nn,-1))
        return new

    def print_network(self, net):  # ,epoch,tau,row1):
        # with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
        # print("------------------------------------------------------ Epoch "+str(epoch)+" ---------------------------------------\n")
        # print("Input row:" +str(row1))
        for i, layer in enumerate(net, 1):
            if (i == 1):
                # print("=============== PreMiddle layer =============== \n")
                # for neuron in (net[0]['Premiddle']):
                #     print("Weights  :" + str(neuron['weights']))
                #     print("delta  :" + str(neuron['delta']))
                #     print("result  :" + str(neuron['result']))
                print("=============== Middle layer =============== \n")
                for neuron in (net[1]['middle']):
                    # print("Weights  :" + str(neuron['weights']))
                    # print("delta  :" + str(neuron['delta']))
                    print("result  :" + str(neuron['result']))
                # print("=============== Postiddle layer =============== \n")
                # for neuron in (net[0]['Postmiddle']):
                #     print("Weights  :" + str(neuron['weights']))
                #     print("delta  :" + str(neuron['delta']))
                #     print("result  :" + str(neuron['result']))
            # else:
            #     print("=============== Output layer =============== \n")
            #     for neuron in (net[1]['output']):
            #         print("Weights  :" + str(neuron['weights']))
            #         print("delta  :" + str(neuron['delta']))
      
      
      
            #         print("result  :" + str(neuron['result']))


    def generate_wt(self,x, y):
        l =[]
        for i in range(x * y):
            l.append(np.random.uniform(low=-0.9, high=0.9))
        aa=(np.array(l).reshape(x, y))
        
        return aa

    def initialize_network(self,InNetInputNons, hiddenlist, outs):
        w1 = self.generate_wt(InNetInputNons, hiddenlist)
        w2 = self.generate_wt(hiddenlist, outs)
        return w1,w2


    def forward_propagation(self, w1,w2, input1, n_outputs, ssteps, scale, dropout):
        cache=[]
        z1 = input1.dot(w1)# input from layer 1
        a1 = self.PSS(z1, ssteps, scale)# out put of layer 2

        z2 = a1.dot(w2)# input of out layer
        a2 = self.PSS(z2, ssteps, scale)# output of out layer

        return a2, cache

    def back_propagation(self, w1,w2,lrate, input1,InNetInputNo, expected, outputs, n_outputs, ssteps, scale, cache,
                        dropout):
    
            z1 = input1.dot(w1)# input from layer 1
            a1 = self.PSS(np.array([z1]), ssteps, scale)# out put of layer 2
            z2 = a1.dot(w2)# input of out layer
            a2 = self.PSS(z2, ssteps, scale)# output of out layer

            d2 = self.DSpearman(a2, np.array([expected]))
            xx=np.multiply(a1, 1-a1)
            ee=(w2.dot((np.array(d2)))).transpose()
            d1 = np.multiply(ee,(xx))

            w1_adj = np.array([input1]).transpose().dot(d1)
            w2_adj = d2.dot(a1)

            w1 = w1-(lrate*(w1_adj[0]))
            w2 = w2-(lrate*(w2_adj))

            return  w1,w2


    def PNNFit(self, w1,w2, epochs, train_fold_features, train_fold_labels, features_no, n_outputs, ssteps, lrate
               , middle, scale):

        statelayer = [0] * middle
        z = len(train_fold_features) + 1
        for epoch in range(epochs):
            rr = 0
            iterationoutput = np.array([])
            for i, row in enumerate((train_fold_features)):
                xxx1 = np.array(list(row))
                trainfoldexpected = train_fold_labels[i]
                outputs, cache = self.forward_propagation(w1,w2, xxx1, n_outputs, ssteps, scale, False)
                w1,w2 = self.back_propagation(w1,w2,lrate,xxx1, features_no, trainfoldexpected, outputs, n_outputs,
                                                    ssteps, scale, cache, False)

                cc = self.calculateoutputTau([outputs, trainfoldexpected])
                iterationoutput = np.append(iterationoutput, cc)
            rr = sum(iterationoutput) / z
            w1,w2 = w1,w2

            print("-- Epoch %d", epoch)
            print("Tau per iteration ==> " + str(rr))
            # self.print_network(net)
        w1,w2 = w1,w2

        arr_2d = np.reshape(outputs, (28, 28))
        plt.imshow(arr_2d, cmap='gray')
        plt.show()

        return w1,w2, outputs

    def calculateoutputTau(self, iterationoutput):
        sum_Tau = 0
        tau, pv = spearmanr(np.array(iterationoutput[1]), iterationoutput[0])
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

    def CrossValidationAvg(self, kfold, foldcounter, foldindex, X_train, y_train, featuresno, middle,
                           labelno, ssteps, lrate, bbs, epochs, bestvector):
        w1,w2 = self.initialize_network(featuresno, middle, labelno)
        avr_res = 0
        tot_etau = 0
        # for idx_train, idx_test in kfold.split(X_train):
        foldindex += 1
        trainlabel = []
        testlabel = []

        w1,w2, error = self.PNNFit(w1,w2, epochs, X_train, y_train, featuresno, labelno, ssteps,
                                 lrate, middle, bbs)
        # self.print_network(net)
        iterationoutput = self.predict(w1,w2, X_train, testlabel, labelno, ssteps, bbs,
                                       middle)
        # self.print_network(net)
        print("-- Predition one fold Result %d", iterationoutput)
        tot_etau += iterationoutput
        avr_res = tot_etau / foldcounter
        print("Final average %.2f Folds test Result %.8f", foldcounter, avr_res)

        return avr_res, w1,w2

    def training(self, epochs, X, y, featuresno, labelno, ssteps, lrate, middle, scale):
        foldcounter = 10
        kfold = sklearn.model_selection.KFold(foldcounter, shuffle=True, random_state=1)
        foldindex = 0
        lrlist = [0.05]  # ,0.09,0.1,0.2,0.3,0.4,0.5]
        scalelist = [scale]
        # hnlist=[hn]
        bestvector = [0, 0, 0, 0, 0]
        avresult = 0
        bestvresult = 0
        # for hn1 in hnlist:
        for lr1 in lrlist:
            for scl in scalelist:
                avresult, bestnet = self.CrossValidationAvg(kfold, foldcounter, foldindex, X, y, featuresno,
                                                            middle, labelno, ssteps, lr1, scl, epochs,
                                                            bestvector)
                print('crossv Prediction=%f , lr=%f', (avresult, lr1))
                if (avresult > bestvresult):
                    bestvresult = avresult
                    bestvector = [bestnet, lr1, middle, bestvresult, scl]
        now = datetime.now()
        timestamp = datetime.timestamp(now)

        return bestnet, avresult

    def predict(self, w1,w2, test_fold_features, test_fold_labels, n_outputs, labelvalue, bx, premiddle, middle,
                postmiddle):
        iterationoutput = np.array([])
        statelayer = list()
        for i, row in enumerate((test_fold_features)):
            xxx1 = np.array(list(row))
            testfoldlabels = list(test_fold_labels[i])
            predicted, statelayer = self.forward_propagation(w1,w2, row, testfoldlabels, n_outputs, labelvalue, bx,
                                                             statelayer)
            iterationoutput = np.append(iterationoutput, [self.calculateoutputTau([predicted, testfoldlabels])])
        avrre = sum(iterationoutput) / len(test_fold_features)
        return avrre

    def loadData1(self, filename, featuresno, labelno, labelvalue, iteration, lrate, middle, ):
        data = list()
        labels = list()
        alldata = list()
        scale = 2 * labelno
        print("==================================" + filename + "=============================")
        filename1 = filename
        gpsTrack = open(filename1, "r")
        csvReader = csv.reader(gpsTrack)
        next(csvReader)
        for row in csvReader:
            data.append(row[0:featuresno])
            labels.append(row[featuresno:featuresno + labelno])
            alldata.append(row[:])

        y = np.array(labels)
        X = np.array(data)
        net1 = self.loadData(X.astype(np.float32), y.astype(np.int32), featuresno, labelno, labelvalue, iteration,
                             lrate, middle, scale)
        print('Done')
        return net1

    def loadData(self, X, y, featuresno, labelno, ssteps, iteration, lrate, middle, scale):
        features_norm = zscore(X, axis=1)
        # features_norm=np.array(X)
        net1, tot_error2 = self.training(iteration, X, y, featuresno, labelno, ssteps, lrate,
                                         middle, scale)
        print('Done')
        return net1
###############################################################
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################