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


class PAOneLayer:

    def __init__(self):
        print("AutoEncoder One Layer Starting..")

    def SSS(self, xi, nlabel, bx):
        sum2 = 0
        s = 100
        b = 100 / bx
        t = 200
        for i in range(nlabel):
            xx = s - ((i * t) / (nlabel - 1))
            sum2 += 0.5 * (np.tanh((-b * (xi)) - (xx)))
        sum2 = -1 * sum2
        sum2 = sum2 + (nlabel * 0.5)
        return sum2

    def dSSS(self, xi, nlabel, bx):
        derivative2 = 0
        s = 100
        b = 100 / bx
        t = 200
        for i in range(nlabel):
            xx = s - ((i * t) / (nlabel - 1))
            derivative2 += 0.5 * (1 - np.power(np.tanh((-b * (xi)) - (xx)), 2))
        derivative2 = -1 * derivative2
        derivative2 = derivative2 + (nlabel * 0.5)
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

    def PSS_Smooth(self, xi, n, stepwidth=2):
        sum1 = 0
        b = 100
        for i in range(n):
            sum1 += -0.5 * (np.tanh(-b * (xi - (stepwidth * i))))
        sum1 = sum1 + (n / 2)
        return sum1

    def dPSS_Smooth(self, xi, n, stepwidth=2):
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
        n = len(output)
        dif = 0
        diflist = np.array([])
        deflist = np.array([])
        for i in range(n):
            o = output[i]
            xx = expected
            e = xx[i]
            diflist = np.append(diflist, [2 * (o - e)])
            dif += 2 * (o - e)

        den = n * (np.power(n, 2) - 1)
        for dd in diflist:
            deflist = np.append(deflist, [((dd) / (den))])
        return deflist

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

    def initialize_network(self, ins, middle, outs):

        input_neurons = ins
        # Pre_neurons = premiddle
        # Post_neurons = postmiddle
        middle_neurons = middle
        output_neurons = outs

        net = list()
        # Premiddle_layer = {
        #     'Premiddle': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=input_neurons)} for i in
        #                   range(Pre_neurons)]}
        # net.append(Premiddle_layer)
        hidden_layer = {'middle': [{'weights': np.random.uniform(low=0, high=0.05, size=input_neurons)} for i in
                                   range(middle_neurons)]}
        net.append(hidden_layer)
        # Postmiddle_layer = {
        #     'Postmiddle': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=middle_neurons)} for i in
        #                    range(Post_neurons)]}
        # net.append(Postmiddle_layer)
        output_layer = {'output': [{'weights': np.random.uniform(low=0, high=0.05, size=middle_neurons)} for i in
                                   range(output_neurons)]}
        net.append(output_layer)

        return net

    def forward_propagation(self, net, input1, trainfold, n_outputs, ssteps, b, statelayer):
        recuurent = False
        row1 = input1
        # prev_input = np.array([])

        # for nindex, neuron in enumerate(net[0]['Premiddle']):
        #     sum1 = neuron['weights'].T.dot(row1)
        #     # if (np.isnan(sum1)):
        #     #     sum1 = 0
        #     result = self.SSS(sum1, ssteps, b)
        #     # if np.isnan(result):
        #     #     result = 0
        #     neuron['result'] = result
        #     prev_input = np.append(prev_input, [result])
        # row1 = prev_input

        prev_input = np.array([])
        for nindex, neuron in enumerate(net[0]['middle']):
            sum1 = neuron['weights'].T.dot(row1)
            # if (np.isnan(sum1)):
            #     sum1 = 0
            result = self.PSS(sum1, ssteps, b)
            # result = sum1
            # if np.isnan(result):
            #     result = 0
            neuron['result'] = result
            neuron['latentcode'] = result
            prev_input = np.append(prev_input, [result])
        row1 = prev_input

        # prev_input = np.array([])
        # for nindex, neuron in enumerate(net[2]['Postmiddle']):
        #     sum1 = neuron['weights'].T.dot(row1)
        #     # if (np.isnan(sum1)):
        #     #     sum1 = 0
        #     result = self.SSS(sum1, ssteps, b)
        #     # if np.isnan(result):
        #     #     result = 0
        #     neuron['result'] = result
        #     prev_input = np.append(prev_input, [result])
        # row1 = prev_input

        statelayer = prev_input

        prev_input = np.array([])
        for neuron in net[1]['output']:
            sum1 = neuron['weights'].T.dot(row1)
            result = self.PSS(sum1, ssteps, b)
            # if np.isnan(result):
            #     result = 0
            neuron['result'] = result
            prev_input = np.append(prev_input, [result])

        row1 = prev_input
        return row1, statelayer

    def back_propagation(self, net, features_no, expected, outputs, n_outputs, ssteps, b):

        results = list()
        errors = np.array([])
        errors = np.array([0] * n_outputs, dtype=float)
        results = [neuron['result'] for neuron in net[1]['output']]
        errors = self.DSpearman(results, expected)

        for indexsub, neuron in enumerate(net[1]['output']):
            neuron['delta'] = errors[indexsub] * self.dPSS(neuron['result'], ssteps, b)
            if (np.isnan(neuron['delta'])):
                neuron['delta'] = 0

        # errors=np.array([0]* len(net[2]['Postmiddle']) , dtype=float)
        # for j, neuron in enumerate(net[2]['Postmiddle']):
        #     herror = 0
        #     for outneuron in (net[3]['output']):
        #         herror += (outneuron['weights'][j]) * (outneuron['delta'])
        #     errors = np.append(errors, [herror])
        # for indexsub, neuron in enumerate(net[2]['Postmiddle']):
        #     neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], ssteps, b)
        #     if (np.isnan(neuron['delta'])):
        #         neuron['delta'] = 0

        errors = np.array([0] * len(net[0]['middle']), dtype=float)
        for j, neuron in enumerate(net[0]['middle']):
            herror = 0
            for outneuron in (net[1]['output']):
                herror += (outneuron['weights'][j]) * (outneuron['delta'])
            errors = np.append(errors, [herror])
        for indexsub, neuron in enumerate(net[0]['middle']):
            neuron['delta'] = errors[indexsub] * self.dPSS(neuron['result'], ssteps, b)
            if (np.isnan(neuron['delta'])):
                neuron['delta'] = 0

        # errors=np.array([0]* len(net[0]['Premiddle']) , dtype=float)
        # for j, neuron in enumerate(net[0]['Premiddle']):
        #     herror = 0
        #     for outneuron in (net[1]['middle']):
        #         herror += (outneuron['weights'][j]) * (outneuron['delta'])
        #     errors = np.append(errors, [herror])
        # for indexsub, neuron in enumerate(net[0]['Premiddle']):
        #     neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], ssteps, b)
        #     if (np.isnan(neuron['delta'])):
        #         neuron['delta'] = 0

    def updateWeights(self, net, input1, lrate):

        inputs = list(input1)

        # for neuron in (net[0]['Premiddle']):
        #     for j in range(len(inputs)):
        #         neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
        #     neuron['weights'][-1] -= lrate * neuron['delta']
        # inputs = [neuron['result'] for neuron in net[0]['Premiddle']]

        for neuron in (net[0]['middle']):
            for j in range(len(inputs)):
                neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= lrate * neuron['delta']
        inputs = [neuron['result'] for neuron in net[0]['middle']]

        # for neuron in (net[2]['Postmiddle']):
        #     for j in range(len(inputs)):
        #         neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
        #     neuron['weights'][-1] -= lrate * neuron['delta']
        # inputs = [neuron['result'] for neuron in net[2]['Postmiddle']]

        for neuron in (net[1]['output']):
            for j in range(len(inputs)):
                neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= lrate * neuron['delta']

        return net, neuron

    def PNNFit(self, net, epochs, train_fold_features, train_fold_labels, features_no, n_outputs, ssteps, lrate
               , middle, b):

        statelayer = [0] * middle
        z = len(train_fold_features) + 1
        for epoch in range(epochs):
            rr = 0
            iterationoutput = np.array([])
            for i, row in enumerate((train_fold_features)):
                xxx1 = np.array(list(row))
                trainfoldexpected = train_fold_labels[i]
                outputs, statelayer = self.forward_propagation(net, xxx1, trainfoldexpected, n_outputs, ssteps, b,
                                                               statelayer)
                self.back_propagation(net, features_no, xxx1, outputs[0], n_outputs, ssteps, b)
                net, neuron = self.updateWeights(net, xxx1, lrate)
                cc = self.calculateoutputTau([outputs, trainfoldexpected])
                iterationoutput = np.append(iterationoutput, cc)
            rr = sum(iterationoutput) / z
            net = net

            print("-- Epoch %d", epoch)
            print("Tau per iteration ==> " + str(rr))
            # self.print_network(net)
        net = net

        arr_2d = np.reshape(outputs, (28, 28))
        plt.imshow(arr_2d, cmap='gray')
        plt.show()

        return net, outputs

    def calculateoutputTau(self, iterationoutput):
        sum_Tau = 0
        tau, pv = spearmanr(np.array(iterationoutput[1]), iterationoutput[0])
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

    def CrossValidationAvg(self, kfold, foldcounter, foldindex, X_train, y_train, featuresno, middle,
                           labelno, ssteps, lrate, bbs, epochs, bestvector):
        net = self.initialize_network(featuresno, middle, labelno)
        avr_res = 0
        tot_etau = 0
        # for idx_train, idx_test in kfold.split(X_train):
        foldindex += 1
        trainlabel = []
        testlabel = []
        # for i in idx_train:
        #     trainlabel.append(y_train[i])
        # for i in idx_test:
        #     testlabel.append(y_train[i])
        net, error = self.PNNFit(net, epochs, X_train, y_train, featuresno, labelno, ssteps,
                                 lrate, middle, bbs)
        # self.print_network(net)
        iterationoutput = self.predict(net, X_train, testlabel, labelno, ssteps, bbs,
                                       middle)
        # self.print_network(net)
        print("-- Predition one fold Result %d", iterationoutput)
        tot_etau += iterationoutput
        avr_res = tot_etau / foldcounter
        print("Final average %.2f Folds test Result %.8f", foldcounter, avr_res)

        return avr_res, net

    def training(self, epochs, X, y, featuresno, labelno, ssteps, lrate, middle, scale):
        foldcounter = 10
        kfold = sklearn.model_selection.KFold(foldcounter, shuffle=True, random_state=1)
        foldindex = 0
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
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

    def predict(self, net1, test_fold_features, test_fold_labels, n_outputs, labelvalue, bx, premiddle, middle,
                postmiddle):
        iterationoutput = np.array([])
        statelayer = list()
        for i, row in enumerate((test_fold_features)):
            xxx1 = np.array(list(row))
            testfoldlabels = list(test_fold_labels[i])
            predicted, statelayer = self.forward_propagation(net1, row, testfoldlabels, n_outputs, labelvalue, bx,
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