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
# from decimal import *
from scipy.stats import zscore
import scipy.stats
from statistics import mean
from itertools import combinations, permutations
import csv
import scipy.stats as ss
# from sympy import *
import random
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import itertools
from scipy.stats.mstats import spearmanr
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime
import time
import pandas as pd
import os.path

data = []
labels = []
alldata = []

class SGPN3:
    ops = {}
    def __init__(self):
        print("SGPN has 3 subgroups Starting..")

    # Error Function#
    def SSS(self,xi, n, boundaryValue):
        sum = 0
        c = 100
        b = 100 / boundaryValue
        for i in range(n):
            sum += -0.5 * (np.tanh((-b * (xi)) - (c * (1 - ((2 * i) / (n - 1))))))
        sum = sum + (n * 0.5)
        return sum


    
    def dSSS(self,xi, nlabel, bx):
        k = (xi, nlabel, bx)
        v = self.ops.get(k)

        if v is None:
            derivative2 = 0
            s = 100
            b = 100 / bx
            t = 200
            for i in range(nlabel):
                xx = s - ((i * t) / (nlabel - 1))
                derivative2 += 0.5 * (1 - np.power(np.tanh((-b * (xi)) - (xx)), 2))
            derivative2 = -1 * derivative2
            derivative2 = derivative2 + (nlabel * 0.5)

            self.ops[k] = derivative2
            return derivative2
        else:
            return v


    def Spearman(self,output, expected):
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


    def DSpearman(self,output, expected):
        n = len(expected)
        dif = 0
        diflist = np.array([])
        deflist = np.array([])
        for i in range(n):
            diflist = np.append(diflist, [2 * (output[i]) - 2 * (expected[i])])
            dif += 2 * (output[i]) - 2 * (expected[i])

        den = n * (np.power(n, 2) - 1)
        for dd in diflist:
            deflist = np.append(deflist, [((dd) / (den))])
        return deflist


    # Drawing image per each iteration for the video#
    def DrawGraph(self,net1, inputs, expected, hn, preverror, epoch, totalerror, totalepoch, epochs, imageindex):
        fig = plt.figure(figsize=self.cm2inch(19, 24))
        fig.suptitle('SGPN Ranking', fontsize=14, fontweight='bold')
        errline = [0] * 500
        for u in range(len(preverror)):
            errline[u] = preverror[u]
        ############################################################
        axes2 = fig.add_axes([0.1, 0.5, 0.3, 0.4])  # inset axes
        axes2.plot(list(range(0, len(preverror))), preverror, 'g')
        axes2.set_xlabel('No. iterations')
        axes2.set_ylabel('Spearman Correlation')
        axes2.set_ylim([-0.2, 1.1])
        axes2.set_title('Avg. Ranking')
        ############################################################
        ax = fig.add_subplot(111)
        #####################################
        input_neurons = len(inputs)
        hidden_neurons = hn
        hidden_layer = net1[0]
        output_layer = net1[1]
        G = nx.Graph()
        for ii in range(input_neurons):
            G.add_node(ii, label=inputs[ii], pos=(0.2, 1.6 + (ii / 1)))
        hh = []
        for j in range(hidden_neurons):
            G.add_node(j + input_neurons, label=self.truncate(hidden_layer[j]['result'], 3), pos=(0.8, 1.2 + (j / 2)))
            for k in range(len(hidden_layer[j]['weights'])):
                hh.append(self.truncate(hidden_layer[j]['weights'][k], 3))
        results = []
        p = 0
        ind = 1
        for k in range(2):
            p += k
            subresults = []
            for j in range(len(expected[k])):
                ind += 1
                p += j
                res = output_layer[k][j]['result']
                subresults.append(self.truncate(res, 3))
                G.add_node(p + input_neurons + hidden_neurons, label=self.truncate(res, 3), pos=(1.4, k + (ind / 2)))
                for w in range(len(output_layer[k][j]['weights'])):
                    hh.append(self.truncate(output_layer[k][j]['weights'][w], 3))
            results.append(subresults)

        node_labels = nx.get_node_attributes(G, 'label')
        pos = nx.get_node_attributes(G, 'pos')
        counter = 0
        for i in range(input_neurons):
            for j in range(hidden_neurons):
                G.add_edge(i, j + input_neurons, label=hh[counter], fontsize=8)
                counter += 1
        edge_labels = nx.get_edge_attributes(G, 'label')
        for i in range(hidden_neurons):
            p = 0
            for k in range(2):
                p += k
                for j in range(len(expected[k])):
                    p += j
                    G.add_edge(i + input_neurons, p + input_neurons + hidden_neurons, label=hh[counter], fontsize=0.05)
                    counter += 1
        edge_labels = nx.get_edge_attributes(G, 'label')
        weights = [G[u][v]['label'] + 1 for u, v in edge_labels]
        nx.draw(G, pos=pos, labels=node_labels, with_labels=True, edge_labels=edge_labels, edge_color='#d3d3d3',
                width=weights, node_size=2000, node_color='#e5e5e5')
        nx.draw_networkx_edge_labels(G, pos, labels=node_labels, with_labels=True, edge_labels=edge_labels, font_size=6,
                                    label_pos=0.25, edge_color='#d3d3d3')
        ax.text(0.7, 6.3, 'L.Rate=0.05', fontsize=12)
        ax.text(0.7, 6.1, 'MAFN=10', fontsize=12)
        ax.text(1, 6.1, 'epoch=' + str(epoch), fontsize=17)

        ax.text(0.15, 0.9, 'Dataset', fontsize=12)
        if (imageindex % 2 == 0):
            ax.text(0.01, 0.7, '[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12)
            ax.text(0.01, 0.5, '[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12, color='red')
            ax.text(0.65, 0.5, "-->" + str(results[0]) + "," + str(results[1]), fontsize=12, color='red')
        else:
            ax.text(0.01, 0.7, '[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12, color='red')
            ax.text(0.65, 0.7, "-->" + str(results[0]) + "," + str(results[1]), fontsize=12, color='red')
            ax.text(0.01, 0.5, '[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12)

        for h in range(2):
            t1, t2 = spearmanr(results[h], expected[h])
            ax.text(1.5, 2.5 + (2 * h), 'Roh' + str(h), fontsize=14)
            ax.text(1.5, 2.2 + (2 * h), str(self.truncate(t1, 2)), fontsize=14)
        ax.text(1.4, 3.3, 'Avg.Roh=' + str(preverror[-1]), fontsize=12)
        # plt.show()
        plt.savefig('C:\\Ayman\\PhDThesis\\video\\test\\' + str(imageindex) + '.png', dpi=150)
        plt.close(fig)
        plt.close('all')
        return G


    def cm2inch(self,*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)


    def calculateoutputTau(self,iterationoutput):
        sum_Tau = 0
        tau1 = 0
        tau2 = 0
        tau3 = 0
        for ii in iterationoutput:
            tau, tauslist = self.calcAvrTau(ii[0], ii[1], 3)
            sum_Tau += tau
            tau1 += tauslist[0]
            tau2 += tauslist[1]
            tau3 += tauslist[2]
        return sum_Tau, tau1, tau2,tau3


    def truncate(self,n, decimals=0):
        v = []
        if type(n) is list:
            for i in range(len(n)):
                v.append(float('%.3f' % n[i]))
        else:
            v = float('%.3f' % n)
        return v


    def createDropNet(self,net):
        keep_prob = 0.5
        layerdrop = []
        for lindex, layer in enumerate(net):
            NeuronDropcache = []
            if (lindex == 0):
                for indexn, neuron in enumerate(layer):
                    xx = neuron['weights']
                    aa = list(xx)
                    NeuronDropcache = list(itertools.chain(NeuronDropcache, aa))
                layerdrop.append(NeuronDropcache)
            else:
                totww = []
                for indnsub, sub1 in enumerate(layer):
                    gweight = []
                    for indexn, neuron in enumerate(sub1):
                        xx = neuron['weights']
                        aa = list(xx)
                        gweight.append(aa)
                    totww.append(gweight)
                layerdrop.append(totww)
        layerdrop1 = []
        for lindex1, ldrop in enumerate(layerdrop):
            if (lindex1 == 0):
                narr = np.array(ldrop)
                NeuronDropcache = []
                D1 = np.random.uniform(low=-0.9, high=0.9, size=narr.size)
                D1 = D1 < keep_prob
                layerdrop1.append(D1.tolist())
            else:
                ddlist = []
                for k in range(len(ldrop)):
                    narr = np.array(ldrop[k])
                    D1 = np.random.uniform(low=-0.9, high=0.9, size=narr.size)
                    D1 = D1 < keep_prob
                    ddlist.append(D1.tolist())
                layerdrop1.append(ddlist)
        return layerdrop1  # ,dropnetperneuron


    def print_network(self,net, epoch, tau, row1, expected):
        my_path = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(my_path, "..//log//SGPN_output.txt")

        with open(filename, 'a') as f:
            print("------------------------------------------------------ Epoch " + str(
                epoch) + " ---------------------------------------\n", file=f)
            print("Input row:" + str(row1) + " Expected:" + str(expected), file=f)
            for i, layer in enumerate(net, 1):
                if (i == 1):
                    print("=============== Middle layer =============== \n", file=f)
                else:
                    print("=============== Output layer =============== \n", file=f)
                for j, neuron in enumerate(layer, 1):
                    print("Subgroup {} :".format(j), neuron, file=f)
            print("==== Roh Correlation = " + str(tau) + "======\n", file=f)


    def initialize_network(self,ins, hiddens, noutputlist):
        input_neurons = ins
        hidden_neurons = hiddens
        n_hidden_layers = 1
        net = list()
        subgroup = list()
        for h in range(n_hidden_layers):
            if h != 0:
                input_neurons = len(net[-1])
                hidden_layers = [
                    {'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=hidden_neurons)} for i
                    in range(hidden_neurons)]
                net.append(hidden_layers)
            else:
                first_layer = [
                    {'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=input_neurons)} for i
                    in range(hidden_neurons)]
                net.append(first_layer)
        for sub in range(len(noutputlist)):
            subgroup.append(
                [{'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=hidden_neurons)} for i in
                range(noutputlist[sub])])
        net.append(subgroup)
        return net


    def forward_propagation(self,net, input1, groupno, noutputlist, noutputvalues, scale):
        start = time.time()
        row1 = [None] * len(noutputlist)
        # if True:
        #     cache = createDropNet(net)
        for k in range(len(noutputlist)):
            row1[k] = np.asarray(input1)
        for index, layer in enumerate(net):
            prev_input = []
            if index == 0:  # MIddle Layer
                for neuron in layer:
                    neuron['result'] = []
                    for A in range(len(noutputlist)):
                        sum1 = neuron['weights'].T.dot(row1[0])
                        result1 = self.SSS(sum1, noutputvalues[A], scale)
                        neuron['result'].append(result1)
                    prev_input.append(neuron['result'])
            else:  # OutputLayer Layer
                outtot = []
                for indexsub, subgroup in enumerate(layer):
                    prev_input = []
                    for neuron in (subgroup):
                        neuron['result'] = []
                        xx = neuron['weights']
                        sum1 = neuron['weights'].T.dot([xx[indexsub] for xx in row1])
                        result1 = self.SSS(sum1, noutputvalues[indexsub], scale)
                        neuron['result'].append(result1)
                        prev_input.append(neuron['result'])
                    prev_input = [j for sub in prev_input for j in sub]
                    outtot.append(prev_input)
            row1 = prev_input
        end = time.time()
        # print('Forwardpropagation time %f', end - start)
        return outtot


    def back_propagation(self,net, row, expected, groupno, noutputlist, noutputvalues, scale, dropout):
        start = time.time()
        for i in reversed(range(len(net))):
            layer = net[i]
            results = list()
            nn = len(layer)
            if i == len(net) - 1:  # output neurons
                results = []
                errors = list()
                for indexsub, subgroup in enumerate(layer):
                    sub_result = [neuron['result'] for neuron in subgroup]
                    results.append(sub_result)
                    output = np.array(results[:][indexsub])
                    errors.append(self.DSpearman(output, expected[indexsub]))

                for indexsub, subgroup in enumerate(layer):
                    for j, neuron in enumerate(subgroup):
                        neuron['delta'] = []
                        a = errors[indexsub][j] * self.dSSS(neuron['result'][0], noutputvalues[indexsub], scale)
                        neuron['delta'].append(a)

            else:
                errors = list()
                for j in range(nn):
                    herror = 0
                    nextlayer = net[i + 1]
                    suberrors = []
                    for indexsub, subgroup in enumerate(nextlayer):
                        # sub_result = []
                        for neuron in subgroup:
                            herror += (neuron['weights'][j] * neuron['delta'][0])
                        suberrors.append(herror)
                    errors.append(suberrors)

                for j in range(nn):
                    neuron = layer[j]
                    neuron['delta'] = []
                    for g in range(groupno):
                        neuron['delta'].append(errors[j][g] * self.dSSS(neuron['result'][g], noutputvalues[g], scale))

        end = time.time()
        # print('back_propagation time %f', end - start)


    def updateWeights(self,net, input1, lratelist, noutputlist, noutputvalues, dropout):
        start = time.time()
        inputs = list()
        nn = len(net)
        for i in range(nn):
            inputs = np.asarray(input1).tolist()
            if i != 0:  # output layer
                inputs = list()
                for h in range(len(noutputlist)):
                    temp1 = [neuron['result'][h] for neuron in net[i - 1]]
                    inputs.append(temp1)
                layer = net[i]
                ni = len(inputs[0])
                for indexsub, subgroup in enumerate(layer):
                    counter = 0
                    for nundex, neuron in enumerate(subgroup):
                        counter += nundex
                        for j in range(ni):
                            neuron['weights'][j] -= lratelist[indexsub] * neuron['delta'][0] * inputs[indexsub][j]
                        neuron['weights'][-1] -= lratelist[indexsub] * neuron['delta'][0]
            else:  # Middle layer
                nni = len(inputs)
                for neuron in (net[i]):
                    for j in range(nni):
                        for h in range(len(noutputlist)):
                            neuron['weights'][j] -= lratelist[h] * neuron['delta'][h] * inputs[j]
                    for h in range(len(noutputlist)):
                        neuron['weights'][-1] -= lratelist[h] * neuron['delta'][h]
        end = time.time()
        # print('updateWeights time %f', end - start)
        return neuron


    def calcAvrTau(self,x_in, y_out, groupno):
        sum_Tau = 0
        doublelist = [0, 0,0]
        # print('#########################')
        for i in range(groupno):
            tau, p_value = spearmanr(x_in[i], y_out[i])
            if not np.isnan(tau):
                sum_Tau += tau
                doublelist[i] += tau

        return sum_Tau / groupno, doublelist


    def plotErrorRate(self,errorRate):
        plt.plot(errorRate)
        plt.ylabel('Error Rate')
        plt.show()


    def plot2GError(self,traing1, traing2, valg1, valg2):
        fig, ax = plt.subplots()
        plt.axes()
        plt.title("Train and Validate model for 2 groups")
        plt.plot(list(range(0, len(traing1))), traing1, label='Train g1', color='blue', marker='o')
        plt.plot(list(range(0, len(traing2))), traing2, label='Train g2', color='black', marker='*')
        plt.plot(list(range(0, len(valg1))), valg1, label='10 fold val. g1', color='green', marker='o')
        plt.plot(list(range(0, len(valg2))), valg2, label='10 fold val. g2', color='red', marker='*')
        plt.xlabel("No. Of iterations ")
        plt.legend()
        plt.ylabel('Roh')
        plt.show()


    def plotTrainValidate(self,trainerror, validateerror):
        fig, ax = plt.subplots()
        plt.axes()
        plt.title("Train-validate")
        plt.plot(list(range(0, len(trainerror))), trainerror, label='Train Error', marker='o')
        plt.plot(list(range(0, len(validateerror))), validateerror, label='Validate Error', marker='o')
        plt.xlabel("No. Of iterations ")
        plt.legend()
        plt.ylabel('Tau')
        plt.show()


    def PNNFit(self,net, train_fold_features, train_fold_labels, noutputlist, noutputvalues, lratelist, epoch, datalength, hn,
            b):
        iterationoutput = list()
        for i, row in enumerate((train_fold_features)):
            xxx1 = list(row)
            trainfoldlbelarray = np.array((train_fold_labels))
            trainfoldexpected = trainfoldlbelarray[i]
            outputs = self.forward_propagation(net, xxx1, len(noutputlist), noutputlist, noutputvalues, b)
            self.back_propagation(net, xxx1, trainfoldexpected, len(noutputlist), noutputlist, noutputvalues, b, True)
            self.updateWeights(net, xxx1, lratelist, noutputlist, noutputvalues, True)
            iterationoutput.append([trainfoldexpected.tolist(), outputs])

        return iterationoutput, net


    def CrossValidationAvg(self,net, kfold, foldindex, n, foldederrorrate, X_train, y_train, featuresno, noofhidden, noutputlist,
                        noutputvalues, groupno, lratelist, bbs, epochs,isFold):
        # net = initialize_network(featuresno, noofhidden, noutputlist)
        aa=0
        bb=0
        if (isFold):
            for idx_train, idx_test in kfold.split(X_train):
                foldindex += 1
                train_fold_features = [X_train[i] for i in idx_train.tolist()]
                train_fold_labels = [y_train[i] for i in idx_train.tolist()]
                test_fold_features = [X_train[i] for i in idx_test.tolist()]
                test_fold_labels = [y_train[i] for i in idx_test.tolist()]

                nn = len(train_fold_features)
                tot_epoch = []
                errorRate_validate = []
                for epoch in range(epochs):
                    iterationoutput_train, net = self.PNNFit(net, train_fold_features, train_fold_labels, noutputlist, noutputvalues,
                                                        lratelist, epoch, n, noofhidden, bbs)
                    sum_Tau_train, tau1, tau2,tau3 = self.calculateoutputTau(iterationoutput_train)
                    tot_epoch.append(epoch)
                    a_avg = tau1 / (nn)
                    b_avg = tau2 / (nn)
                    c_avg = tau3 / (nn)
                    epochError_train = sum_Tau_train / (nn)
                    errorRate_validate.append(epochError_train)
                    if epoch % 1 == 0:
                        print('Epoch result >Epoch=%4d ,Tau=%.4f,Tau_g1=%.4f,Tau_g2=%.4f,Tau_g3=%.4f,' % (
                        epoch, epochError_train, a_avg, b_avg,c_avg))
                foldederrorrate = np.append(foldederrorrate, [sum(errorRate_validate) / len(errorRate_validate)])
                self.predict(test_fold_features, test_fold_labels, net, noutputlist, noutputvalues, bbs, groupno, True)
        else:
                nn = len(X_train)
                tot_epoch = []
                errorRate_validate = []
                for epoch in range(epochs):
                    iterationoutput_train, net = self.PNNFit(net, X_train, y_train, noutputlist, noutputvalues,
                                                        lratelist, epoch, n, noofhidden, bbs)
                    sum_Tau_train, tau1, tau2,tau3 = self.calculateoutputTau(iterationoutput_train)
                    tot_epoch.append(epoch)
                    a_avg = tau1 / (nn)
                    b_avg = tau2 / (nn)
                    c_avg = tau3 / (nn)
                    epochError_train = sum_Tau_train / (nn)
                    errorRate_validate.append(epochError_train)
                    if epoch % 1 == 0:
                        print('Epoch result >Epoch=%4d ,Tau=%.4f,Tau_g1=%.4f,Tau_g2=%.4f,Tau_g3=%.4f,' % (
                        epoch, epochError_train, a_avg, b_avg, c_avg))
                # foldederrorrate = np.append(foldederrorrate, [sum(errorRate_validate) / len(errorRate_validate)])
                # aa, bb = predict(test_fold_features, test_fold_labels, net, noutputlist, noutputvalues, bbs, groupno, True)

        return net


    def training(self,net, epochs, X, y, X_test, y_test, featuresno, noutputlist, groupno, lratelist, noutputvalues, hn, scale,
                fold):
        isFold=False 
        kfold=[]        
        if fold>1:
            kfold = KFold(fold, True, 1)
            isFold=True
        foldindex = 0
        n = len(alldata)
        foldederrorrate = np.array([])
        lrlist = [0.05]
        avresult = 0
        for lr1 in lrlist:
            bestnet= self.CrossValidationAvg(net, kfold, foldindex, n, foldederrorrate, X, y,
                                                featuresno, hn, noutputlist, noutputvalues, groupno, lratelist,
                                                scale, epochs,isFold)
        print(">>>>>>>>Testing data result<<<<<<<<<")
        X_test_norm = zscore(X_test, axis=0)
        self.predict(X_test_norm, y_test, bestnet, noutputlist, noutputvalues, scale, groupno, False)
        print(">>>>>>>>>>>>>>>>>>>>>>>>Good Bye<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return bestnet


    def predict(self,test_fold_features, test_fold_labels, net, noutputlist, noutputvalues, bx, groupno, isvalidate):
        testlist = []
        ns = len(noutputlist)
        for i, row in enumerate((test_fold_features)):
            xxx1 = list(row)
            predicted = self.forward_propagation(net, xxx1, ns, noutputlist, noutputvalues, bx)
            singleTau, Tau_list = self.calcAvrTau(test_fold_labels[i], predicted, groupno)
            testlist.append(Tau_list)
        xxx = [i[0] for i in testlist]
        aa = sum(xxx) / len(testlist)
        yyy = [i[1] for i in testlist]
        bb = sum(yyy) / len(testlist)
        zzz = [i[2] for i in testlist]
        cc = sum(zzz) / len(testlist)
        print("============================================")
        if (isvalidate):
            print('Validate Prediction==>>Tau1=' + str(aa) + ' Tau2=' + str(bb)  +' Tau3='+str(cc))
        else:
            print('Final Prediction==>>Tau1=' + str(aa) + ' Tau2=' + str(bb)+ ' Tau3=' + str(cc))
        print("============================================")
        return 


    def loadTestingData(self,filename, featuresno, noutputlist):
        data = list()
        labels = []
        data = pd.read_csv(filename)
        # data_sampled = data.sample(n=sample, random_state=1)
        X_sampled = data[data.columns[:-4]].to_numpy()
        labels = data[data.columns[-4:]].to_numpy()
        labelssub = [[[i[0], i[1]], [i[2], i[3]]] for i in labels]
        return X_sampled, labelssub

    def rescale(self,values,featuresno,data_no, new_min , new_max ):
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

    def loadData(self,filename, featuresno, noutputlist, epochs, lratelist, noutputvalues, hn, scale, fold):
        Trainingfilename = filename
        gpsTrack = open(Trainingfilename, "r")
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
    # train_features_list_norm = rescale(data,featuresno,data_no,-scale,scale) 
        train_features_list_norm = zscore(data, axis=0)   


        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_features_list_norm, labels, test_size=0.2, random_state=1)
        net = self.initialize_network(ins=featuresno, hiddens=hn, noutputlist=noutputlist)
        #########################################
        net = self.training(net=net, epochs=epochs, X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                            featuresno=featuresno,
                            noutputlist=noutputlist, groupno=groupno, lratelist=lratelist,
                            noutputvalues=noutputvalues, hn=hn, scale=scale, fold=fold)

        return net


class SGPN2:
    
    ops = {}

    def __init__(self):
        print("SGPN has 2 subgroups Starting..")

    def SSS(self,xi, n, boundaryValue):
        sum1 = 0
        c = 100
        b = 100 / boundaryValue
        for i in range(n):
            sum1 += -0.5 * (np.tanh((-b * (xi)) - (c * (1 - ((2 * i) / (n - 1))))))
        sum1 = sum1 + (n * 0.5)
        return sum1


    
    def dSSS(self,xi, nlabel, bx):
        k = (xi, nlabel, bx)
        v = self.ops.get(k)

        if v is None:
            derivative2 = 0
            s = 100
            b = 100 / bx
            t = 200
            for i in range(nlabel):
                xx = s - ((i * t) / (nlabel - 1))
                derivative2 += 0.5 * (1 - np.power(np.tanh((-b * (xi)) - (xx)), 2))
            derivative2 = -1 * derivative2
            derivative2 = derivative2 + (nlabel * 0.5)

            self.ops[k] = derivative2
            return derivative2
        else:
            return v

    def Spearman(self,output, expected):
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


    def DSpearman(self,output, expected):
        n = len(expected)
        dif = 0
        diflist = np.array([])
        deflist = np.array([])
        for i in range(n):
            diflist = np.append(diflist, [2 * (output[i]) - 2 * (expected[i])])
            dif += 2 * (output[i]) - 2 * (expected[i])

        den = n * (np.power(n, 2) - 1)
        for dd in diflist:
            deflist = np.append(deflist, [((dd) / (den))])
        return deflist


    # Drawing image per each iteration for the video#
    def DrawGraph(self,net1, inputs, expected, hn, preverror, epoch, totalerror, totalepoch, epochs, imageindex):
        fig = plt.figure(figsize=self.cm2inch(19, 24))
        fig.suptitle('SGPN Ranking', fontsize=14, fontweight='bold')
        errline = [0] * 500
        for u in range(len(preverror)):
            errline[u] = preverror[u]
        ############################################################
        axes2 = fig.add_axes([0.1, 0.5, 0.3, 0.4])  # inset axes
        axes2.plot(list(range(0, len(preverror))), preverror, 'g')
        axes2.set_xlabel('No. iterations')
        axes2.set_ylabel('Spearman Correlation')
        axes2.set_ylim([-0.2, 1.1])
        axes2.set_title('Avg. Ranking')
        ############################################################
        ax = fig.add_subplot(111)
        #####################################
        input_neurons = len(inputs)
        hidden_neurons = hn
        hidden_layer = net1[0]
        output_layer = net1[1]
        G = nx.Graph()
        for ii in range(input_neurons):
            G.add_node(ii, label=inputs[ii], pos=(0.2, 1.6 + (ii / 1)))
        hh = []
        for j in range(hidden_neurons):
            G.add_node(j + input_neurons, label=self.truncate(hidden_layer[j]['result'], 3), pos=(0.8, 1.2 + (j / 2)))
            for k in range(len(hidden_layer[j]['weights'])):
                hh.append(self.truncate(hidden_layer[j]['weights'][k], 3))
        results = []
        p = 0
        ind = 1
        for k in range(2):
            p += k
            subresults = []
            for j in range(len(expected[k])):
                ind += 1
                p += j
                res = output_layer[k][j]['result']
                subresults.append(self.truncate(res, 3))
                G.add_node(p + input_neurons + hidden_neurons, label=self.truncate(res, 3), pos=(1.4, k + (ind / 2)))
                for w in range(len(output_layer[k][j]['weights'])):
                    hh.append(self.truncate(output_layer[k][j]['weights'][w], 3))
            results.append(subresults)

        node_labels = nx.get_node_attributes(G, 'label')
        pos = nx.get_node_attributes(G, 'pos')
        counter = 0
        for i in range(input_neurons):
            for j in range(hidden_neurons):
                G.add_edge(i, j + input_neurons, label=hh[counter], fontsize=8)
                counter += 1
        edge_labels = nx.get_edge_attributes(G, 'label')
        for i in range(hidden_neurons):
            p = 0
            for k in range(2):
                p += k
                for j in range(len(expected[k])):
                    p += j
                    G.add_edge(i + input_neurons, p + input_neurons + hidden_neurons, label=hh[counter], fontsize=0.05)
                    counter += 1
        edge_labels = nx.get_edge_attributes(G, 'label')
        weights = [G[u][v]['label'] + 1 for u, v in edge_labels]
        nx.draw(G, pos=pos, labels=node_labels, with_labels=True, edge_labels=edge_labels, edge_color='#d3d3d3',
                width=weights, node_size=2000, node_color='#e5e5e5')
        nx.draw_networkx_edge_labels(G, pos, labels=node_labels, with_labels=True, edge_labels=edge_labels, font_size=6,
                                    label_pos=0.25, edge_color='#d3d3d3')
        ax.text(0.7, 6.3, 'L.Rate=0.05', fontsize=12)
        ax.text(0.7, 6.1, 'MAFN=10', fontsize=12)
        ax.text(1, 6.1, 'epoch=' + str(epoch), fontsize=17)

        ax.text(0.15, 0.9, 'Dataset', fontsize=12)
        if (imageindex % 2 == 0):
            ax.text(0.01, 0.7, '[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12)
            ax.text(0.01, 0.5, '[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12, color='red')
            ax.text(0.65, 0.5, "-->" + str(results[0]) + "," + str(results[1]), fontsize=12, color='red')
        else:
            ax.text(0.01, 0.7, '[0.1,0.2,0.3] -- [[1,2,3,4],[3,2,1]]', fontsize=12, color='red')
            ax.text(0.65, 0.7, "-->" + str(results[0]) + "," + str(results[1]), fontsize=12, color='red')
            ax.text(0.01, 0.5, '[0.8,0.9,0.7] -- [[4,2,3,1],[1,3,2]]', fontsize=12)

        for h in range(2):
            t1, t2 = spearmanr(results[h], expected[h])
            ax.text(1.5, 2.5 + (2 * h), 'Roh' + str(h), fontsize=14)
            ax.text(1.5, 2.2 + (2 * h), str(truncate(t1, 2)), fontsize=14)
        ax.text(1.4, 3.3, 'Avg.Roh=' + str(preverror[-1]), fontsize=12)
        # plt.show()
        plt.savefig('C:\\Ayman\\PhDThesis\\video\\test\\' + str(imageindex) + '.png', dpi=150)
        plt.close(fig)
        plt.close('all')
        return G


    def cm2inch(self,*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)


    def calculateoutputTau(self,iterationoutput):
        sum_Tau = 0
        tau1 = 0
        tau2 = 0
        for ii in iterationoutput:
            tau, tauslist = self.calcAvrTau(ii[0], ii[1], 2)
            sum_Tau += tau
            tau1 += tauslist[0]
            tau2 += tauslist[1]
        return sum_Tau, tau1, tau2


    def truncate(self,n, decimals=0):
        v = []
        if type(n) is list:
            for i in range(len(n)):
                v.append(float('%.3f' % n[i]))
        else:
            v = float('%.3f' % n)
        return v


    def createDropNet(self,net):
        keep_prob = 0.5
        layerdrop = []
        for lindex, layer in enumerate(net):
            NeuronDropcache = []
            if (lindex == 0):
                for indexn, neuron in enumerate(layer):
                    xx = neuron['weights']
                    aa = list(xx)
                    NeuronDropcache = list(itertools.chain(NeuronDropcache, aa))
                layerdrop.append(NeuronDropcache)
            else:
                totww = []
                for indnsub, sub1 in enumerate(layer):
                    gweight = []
                    for indexn, neuron in enumerate(sub1):
                        xx = neuron['weights']
                        aa = list(xx)
                        gweight.append(aa)
                    totww.append(gweight)
                layerdrop.append(totww)
        layerdrop1 = []
        for lindex1, ldrop in enumerate(layerdrop):
            if (lindex1 == 0):
                narr = np.array(ldrop)
                NeuronDropcache = []
                D1 = np.random.uniform(low=-0.9, high=0.9, size=narr.size)
                D1 = D1 < keep_prob
                layerdrop1.append(D1.tolist())
            else:
                ddlist = []
                for k in range(len(ldrop)):
                    narr = np.array(ldrop[k])
                    D1 = np.random.uniform(low=-0.9, high=0.9, size=narr.size)
                    D1 = D1 < keep_prob
                    ddlist.append(D1.tolist())
                layerdrop1.append(ddlist)
        return layerdrop1  # ,dropnetperneuron


    def print_network(self,selfnet, epoch, tau, row1, expected):
        with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
            print("------------------------------------------------------ Epoch " + str(
                epoch) + " ---------------------------------------\n", file=f)
            print("Input row:" + str(row1) + " Expected:" + str(expected), file=f)
            for i, layer in enumerate(net, 1):
                if (i == 1):
                    print("=============== Middle layer =============== \n", file=f)
                else:
                    print("=============== Output layer =============== \n", file=f)
                for j, neuron in enumerate(layer, 1):
                    print("Subgroup {} :".format(j), neuron, file=f)
            print("==== Roh Correlation = " + str(tau) + "======\n", file=f)


    def initialize_network(self,ins, hiddens, noutputlist):
        input_neurons = ins
        hidden_neurons = hiddens
        n_hidden_layers = 1
        net = list()
        subgroup = list()
        for h in range(n_hidden_layers):
            if h != 0:
                input_neurons = len(net[-1])
                hidden_layers = [
                    {'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=hidden_neurons)} for i
                    in range(hidden_neurons)]
                net.append(hidden_layers)
            else:
                first_layer = [
                    {'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=input_neurons)} for i
                    in range(hidden_neurons)]
                net.append(first_layer)
        for sub in range(len(noutputlist)):
            subgroup.append(
                [{'delta': [], 'result': [], 'weights': np.random.uniform(low=-0.5, high=0.5, size=hidden_neurons)} for i in
                range(noutputlist[sub])])
        net.append(subgroup)
        return net


    def forward_propagation(self,net, input1, groupno, noutputlist, noutputvalues, scale):
        start = time.time()
        row1 = [None] * len(noutputlist)
        # if True:
        #     cache = createDropNet(net)
        for k in range(len(noutputlist)):
            row1[k] = np.asarray(input1)
        for index, layer in enumerate(net):
            prev_input = []
            if index == 0:  # MIddle Layer
                for neuron in layer:
                    neuron['result'] = []
                    for A in range(len(noutputlist)):
                        sum1 = neuron['weights'].T.dot(row1[0])
                        result1 = self.SSS(sum1, noutputvalues[A], scale)
                        neuron['result'].append(result1)
                    prev_input.append(neuron['result'])
            else:  # OutputLayer Layer
                outtot = []
                for indexsub, subgroup in enumerate(layer):
                    prev_input = []
                    for neuron in (subgroup):
                        neuron['result'] = []
                        xx = neuron['weights']
                        sum1 = neuron['weights'].T.dot([xx[indexsub] for xx in row1])
                        result1 = self.SSS(sum1, noutputvalues[indexsub], scale)
                        neuron['result'].append(result1)
                        prev_input.append(neuron['result'])
                    prev_input = [j for sub in prev_input for j in sub]
                    outtot.append(prev_input)
            row1 = prev_input
        end = time.time()
        # print('Forwardpropagation time %f', end - start)
        return outtot


    def back_propagation(self,net, row, expected, groupno, noutputlist, noutputvalues, scale, dropout):
        start = time.time()
        for i in reversed(range(len(net))):
            layer = net[i]
            results = list()
            nn = len(layer)
            if i == len(net) - 1:  # output neurons
                results = []
                errors = list()
                for indexsub, subgroup in enumerate(layer):
                    sub_result = [neuron['result'] for neuron in subgroup]
                    results.append(sub_result)
                    output = np.array(results[:][indexsub])
                    errors.append(self.DSpearman(output, expected[indexsub]))

                for indexsub, subgroup in enumerate(layer):
                    for j, neuron in enumerate(subgroup):
                        neuron['delta'] = []
                        a = errors[indexsub][j] * self.dSSS(neuron['result'][0], noutputvalues[indexsub], scale)
                        neuron['delta'].append(a)

            else:
                errors = list()
                for j in range(nn):
                    herror = 0
                    nextlayer = net[i + 1]
                    suberrors = []
                    for indexsub, subgroup in enumerate(nextlayer):
                        # sub_result = []
                        for neuron in subgroup:
                            herror += (neuron['weights'][j] * neuron['delta'][0])
                        suberrors.append(herror)
                    errors.append(suberrors)

                for j in range(nn):
                    neuron = layer[j]
                    neuron['delta'] = []
                    for g in range(groupno):
                        neuron['delta'].append(errors[j][g] * self.dSSS(neuron['result'][g], noutputvalues[g], scale))

        end = time.time()
        # print('back_propagation time %f', end - start)


    def updateWeights(self,net, input1, lratelist, noutputlist, noutputvalues, dropout):
        start = time.time()
        inputs = list()
        nn = len(net)
        for i in range(nn):
            inputs = np.asarray(input1).tolist()
            if i != 0:  # output layer
                inputs = list()
                for h in range(len(noutputlist)):
                    temp1 = [neuron['result'][h] for neuron in net[i - 1]]
                    inputs.append(temp1)
                layer = net[i]
                ni = len(inputs[0])
                for indexsub, subgroup in enumerate(layer):
                    counter = 0
                    for nundex, neuron in enumerate(subgroup):
                        counter += nundex
                        for j in range(ni):
                            neuron['weights'][j] -= lratelist[indexsub] * neuron['delta'][0] * inputs[indexsub][j]
                        neuron['weights'][-1] -= lratelist[indexsub] * neuron['delta'][0]
            else:  # Middle layer
                nni = len(inputs)
                for neuron in (net[i]):
                    for j in range(nni):
                        for h in range(len(noutputlist)):
                            neuron['weights'][j] -= lratelist[h] * neuron['delta'][h] * inputs[j]
                    for h in range(len(noutputlist)):
                        neuron['weights'][-1] -= lratelist[h] * neuron['delta'][h]
        end = time.time()
        # print('updateWeights time %f', end - start)
        return neuron


    def calcAvrTau(self,x_in, y_out, groupno):
        sum_Tau = 0
        doublelist = [0, 0]
        # print('#########################')
        for i in range(groupno):
            tau, p_value = spearmanr(x_in[i], y_out[i])
            if not np.isnan(tau):
                sum_Tau += tau
                doublelist[i] += tau

        return sum_Tau / groupno, doublelist


    def plotErrorRate(self,errorRate):
        plt.plot(errorRate)
        plt.ylabel('Error Rate')
        plt.show()


    def plot2GError(self,traing1, traing2, valg1, valg2):
        fig, ax = plt.subplots()
        plt.axes()
        plt.title("Train and Validate model for 2 groups")
        plt.plot(list(range(0, len(traing1))), traing1, label='Train g1', color='blue', marker='o')
        plt.plot(list(range(0, len(traing2))), traing2, label='Train g2', color='black', marker='*')
        plt.plot(list(range(0, len(valg1))), valg1, label='10 fold val. g1', color='green', marker='o')
        plt.plot(list(range(0, len(valg2))), valg2, label='10 fold val. g2', color='red', marker='*')
        plt.xlabel("No. Of iterations ")
        plt.legend()
        plt.ylabel('Roh')
        plt.show()


    def plotTrainValidate(self,trainerror, validateerror):
        fig, ax = plt.subplots()
        plt.axes()
        plt.title("Train-validate")
        plt.plot(list(range(0, len(trainerror))), trainerror, label='Train Error', marker='o')
        plt.plot(list(range(0, len(validateerror))), validateerror, label='Validate Error', marker='o')
        plt.xlabel("No. Of iterations ")
        plt.legend()
        plt.ylabel('Tau')
        plt.show()


    def PNNFit(self,net, train_fold_features, train_fold_labels, noutputlist, noutputvalues, lratelist, epoch, datalength, hn, b):
        iterationoutput = list()
        for i, row in enumerate((train_fold_features)):
            xxx1 = list(row)
            trainfoldlbelarray = np.array((train_fold_labels))
            trainfoldexpected = trainfoldlbelarray[i]
            outputs = self.forward_propagation(net, xxx1, len(noutputlist), noutputlist, noutputvalues, b)
            self.back_propagation(net, xxx1, trainfoldexpected, len(noutputlist), noutputlist, noutputvalues, b, True)
            self.updateWeights(net, xxx1, lratelist, noutputlist, noutputvalues, True)
            iterationoutput.append([trainfoldexpected.tolist(), outputs])

        return iterationoutput, net


    def CrossValidationAvg(self,net, kfold, foldindex, n, foldederrorrate, X_train, y_train, featuresno, noofhidden, noutputlist,
                        noutputvalues, groupno, lratelist, bbs, epochs,isFold):
        aa=0
        bb=0
        if (isFold):
            for idx_train, idx_test in kfold.split(X_train):
                foldindex += 1
                train_fold_features = [X_train[i] for i in idx_train.tolist()]
                train_fold_labels = [y_train[i] for i in idx_train.tolist()]
                test_fold_features = [X_train[i] for i in idx_test.tolist()]
                test_fold_labels = [y_train[i] for i in idx_test.tolist()]

                nn = len(train_fold_features)
                tot_epoch = []
                errorRate_validate = []
                for epoch in range(epochs):
                    iterationoutput_train, net = self.PNNFit(net, train_fold_features, train_fold_labels, noutputlist, noutputvalues,
                                                        lratelist, epoch, n, noofhidden, bbs)
                    sum_Tau_train, tau1, tau2 = self.calculateoutputTau(iterationoutput_train)
                    tot_epoch.append(epoch)
                    a_avg = tau1 / (nn)
                    b_avg = tau2 / (nn)
                    epochError_train = sum_Tau_train / (nn)
                    errorRate_validate.append(epochError_train)
                    if epoch % 1 == 0:
                        print('Epoch result >Epoch=%4d ,Tau=%.4f,Tau_g1=%.4f,Tau_g2=%.4f,' % (
                        epoch, epochError_train, a_avg, b_avg))
                foldederrorrate = np.append(foldederrorrate, [sum(errorRate_validate) / len(errorRate_validate)])
                self.predict(test_fold_features, test_fold_labels, net, noutputlist, noutputvalues, bbs, groupno, True)
        else:
                nn = len(X_train)
                tot_epoch = []
                errorRate_validate = []
                for epoch in range(epochs):
                    iterationoutput_train, net = self.PNNFit(net, X_train, y_train, noutputlist, noutputvalues,
                                                        lratelist, epoch, n, noofhidden, bbs)
                    sum_Tau_train, tau1, tau2 = self.calculateoutputTau(iterationoutput_train)
                    tot_epoch.append(epoch)
                    a_avg = tau1 / (nn)
                    b_avg = tau2 / (nn)
                    epochError_train = sum_Tau_train / (nn)
                    errorRate_validate.append(epochError_train)
                    if epoch % 1 == 0:
                        print('Epoch result >Epoch=%4d ,Tau=%.4f,Tau_g1=%.4f,Tau_g2=%.4f,' % (
                        epoch, epochError_train, a_avg, b_avg))

        return net


    def training(self,net, epochs, X, y, X_test, y_test, featuresno, noutputlist, groupno, lratelist, noutputvalues, hn, scale,
                fold):
        isFold=False 
        kfold=[]        
        if fold>1:
            kfold = KFold(fold, True, 1)
            isFold=True
        foldindex = 0
        n = len(alldata)
        foldederrorrate = np.array([])
        lrlist = [0.05]
        avresult = 0
        for lr1 in lrlist:
            bestnet= self.CrossValidationAvg(net, kfold, foldindex, n, foldederrorrate, X, y,
                                                featuresno, hn, noutputlist, noutputvalues, groupno, lratelist,
                                                scale, epochs,isFold)
        print(">>>>>>>>Testing data result<<<<<<<<<")
        X_test_norm = zscore(X_test, axis=0)
        predict(X_test_norm, y_test, bestnet, noutputlist, noutputvalues, scale, groupno, False)
        print(">>>>>>>>>>>>>>>>>>>>>>>>Good Bye<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return bestnet


    def predict(self,test_fold_features, test_fold_labels, net, noutputlist, noutputvalues, bx, groupno, isvalidate):
        testlist = []
        ns = len(noutputlist)
        for i, row in enumerate((test_fold_features)):
            xxx1 = list(row)
            predicted = self.forward_propagation(net, xxx1, ns, noutputlist, noutputvalues, bx)
            singleTau, Tau_list = self.calcAvrTau(test_fold_labels[i], predicted, groupno)
            testlist.append(Tau_list)
        xxx = [i[0] for i in testlist]
        aa = sum(xxx) / len(testlist)
        yyy = [i[1] for i in testlist]
        bb = sum(yyy) / len(testlist)
        print("============================================")
        if (isvalidate):
            print('Validate Prediction==>>Tau1=' + str(aa) + ' Tau2=' + str(bb))  # +' Tau3='+str(cc))
        else:
            print('Final Prediction==>>Tau1=' + str(aa) + ' Tau2=' + str(bb))
        print("============================================")
        return 


    def loadTestingData(self,filename, featuresno, noutputlist):
        data = list()
        labels = []
        data = pd.read_csv(filename)
        # data_sampled = data.sample(n=sample, random_state=1)
        X_sampled = data[data.columns[:-4]].to_numpy()
        labels = data[data.columns[-4:]].to_numpy()
        labelssub = [[[i[0], i[1]], [i[2], i[3]]] for i in labels]
        return X_sampled, labelssub

    def rescale(self,values,featuresno,data_no, new_min , new_max ):
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

    def loadData(self,filename, featuresno, noutputlist, epochs, lratelist, noutputvalues, hn, scale, fold):

        Trainingfilename = filename
        gpsTrack = open(Trainingfilename, "r")
        csvReader = csv.reader(gpsTrack)
        groupno=len(noutputlist)
        data = list()
        labels=[]
        next(csvReader)
        labelno1=noutputlist[0]
        labelno2=noutputlist[1]
        for row in csvReader:
            data.append(list(map(float,row[0:featuresno])))
            a=list(map(float,row[featuresno:featuresno + labelno1]))
            b=list(map(float,row[featuresno+labelno1:featuresno + labelno1+labelno2]))
            ######################################
            ######################################
            labels.append([a,b])#,c])#,c,d])
        data_no=len(labels)
    # train_features_list_norm = rescale(data,featuresno,data_no,-scale,scale) 
        train_features_list_norm = zscore(data, axis=0)   


        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_features_list_norm, labels, test_size=0.2, random_state=1)
        net = self.initialize_network(ins=featuresno, hiddens=hn, noutputlist=noutputlist)
        #########################################
        net = self.training(net=net, epochs=epochs, X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                            featuresno=featuresno,
                            noutputlist=noutputlist, groupno=groupno, lratelist=lratelist,
                            noutputvalues=noutputvalues, hn=hn, scale=scale, fold=fold)

        return net


    ###############################################################################################################################
    ###############################################################################################################################
    ###############################################################################################################################

    ##########################################################################################