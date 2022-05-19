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
from scipy.stats.mstats import spearmanr
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import itertools
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
# import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime
# from keras.datasets import mnist
# from keras.datasets import cifar10
import pandas as pd
# from keras.utils import np_utils
# from extra_keras_datasets import emnist
# from keras.datasets import fashion_mnist
# from statistics import mean
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from tensorflow.keras.utils import to_categorical
# import pickle
# import logging
# from multiprocessing import Process, Value, Array
# from sklearn import preprocessing
# import multiprocessing
# from multiprocessing import Manager, Pool
# import threading
# from threading import *
    # python >= 3.8
# from multiprocessing.managers import SharedMemoryManager as Manager


# manager = multiprocessing.Manager()

featuresize=28
labelno=2
labelvalue=2
lrate=0.001
noNetwork = 3
kernelSizeList = [5 , 10, 15]
hnnolist =  [50, 100, 150]

def init_outputlayer(hiddenlist,output_neurons):
        outputnet=[]
        sumhn = sum(hiddenlist)
        output_layer = {
            'output': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=sumhn)} for i in range(output_neurons)]}
        outputnet.append(output_layer)
        return outputnet

class PNN:
    
    def __init__(self,start = 0):
        print("PNN Starting..")
        # self.lock = threading.Lock()
        self.value = start

    def PSS(self, xi, n, stepwidth=2):
        sum = 0
        b = 100
        for i in range(n):
            sum += -0.5 * (np.tanh(-b * (xi - (stepwidth * i))))
        sum = sum + (n / 2)
        return sum

    def dPSS(self, xi, n, stepwidth=2):
        sum = 0
        b = 100
        for i in range(n):
            sum += -0.5 * (1 - np.power(np.tanh(-b * (xi - (stepwidth * i))), 2))
        sum = sum + (n / 2)
        return sum

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1. - x)

    def SSS(self, xi, n, boundaryValue):
        sum = 0
        c = 100
        b = 100 // boundaryValue
        for i in range(n):
            sum += -0.5 * (np.tanh((-b * (xi)) - (c * (1 - ((2 * i) // (n + 0.0001 - 1))))))
        sum = sum + (n * 0.5)
        return sum

    def dSSS(self, xi, nlabel, bx):
        derivative2 = 0
        s = 100
        b = 100 // bx
        t = 200
        for i in range(nlabel):
            xx = s - ((i * t) // ((nlabel + 0.0001) - 1))
            derivative2 += 0.5 * (1 - np.power(np.tanh((-b * (xi)) - (xx)), 2))
        derivative2 = -1 * derivative2
        derivative2 = derivative2 + (nlabel * 0.5)
        return derivative2

    def DSpearmanImage(self, output, droh):
        n = len(output)
        n_cu_min_n = float(n ** 3 - n) / -6
        rhs = droh * n_cu_min_n
        diflist2 = [(2 * (rhs - i)) for i in output]
        return np.array(diflist2)

    def DSpearman(self, output, expected):
        n = len(expected)
        diflist = [2 * (output[i] - int(expected[i])) for i in range(n)]
        return diflist



    def initialize_network(self, iInNet, InNetInputNons, hiddenlist, outs):
        output_neurons = outs
        net = list()
        sumhn = sum(hiddenlist)

        for inp in range(iInNet):
            input_layer = {'input': [{'result': 0} for i in range((InNetInputNons[inp]// 2))]}
            net.append(input_layer)

        for inp in range(iInNet):
            hidden_layer = {
                'middle': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=(InNetInputNons[inp]// 2))}
                        for i in range(hiddenlist[inp])]}
            net.append(hidden_layer)

        # output_layer = {
        #     'output': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=sumhn)} for i in range(output_neurons)]}
        # net.append(output_layer)

        return net

    def forward_propagation(self, net,output_net, input1, n_outputs, labelvalue, scale, noNet):
    
        for ind, row1 in enumerate(input1):
            for indx, neuron in enumerate(net[ind]['input']):
                neuron['result'] = row1[indx]
        prev_input = np.array([])
        for ind, row1 in enumerate(input1):
            for neuron in net[ind + noNet]['middle']:
                sum1 = neuron['weights'].T.dot(row1)
                result = self.SSS(sum1, labelvalue, scale)
                neuron['result'] = result
                prev_input = np.append(prev_input, [result])
            row1 = prev_input
        # print('Waiting for a lock')
        prev_input = np.array([])
        for neuron in output_net[0]['output']:
            sum1 = neuron['weights'].T.dot(row1)
            result = self.SSS(sum1, labelvalue, scale)
            neuron['result'] = result
            prev_input = np.append(prev_input, [result])

        row1 = prev_input
  
        return row1

    def back_propagation(self, net,output_net, InNetInputNo, expected, outputs, n_outputs, labelvalue, scale, noNet):
        results = np.array(n_outputs)
        errors = np.array([])
        # print('Waiting for a lock \\n')
        # self.lock.acquire()
        # try:
        results = [neuron['result'] for neuron in output_net[0]['output']]
        errors = self.DSpearman(results, expected)
        # finally:
            # print('Released a lock \\n')
            # self.lock. release()  
        for indexsub, neuron in enumerate(output_net[0]['output']):
            neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
            if (np.isnan(neuron['delta'])):
                neuron['delta'] = 0
        for ind in range(noNet):
            for j, neuron in enumerate(net[ind + noNet]['middle']):
                herror = 0
                for outneuron in (output_net[0]['output']):
                    herror += (outneuron['weights'][j]) * (outneuron['delta'])
                errors = np.append(errors, [herror])

            for indexsub, neuron in enumerate(net[ind + noNet]['middle']):
                neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
                if (np.isnan(neuron['delta'])):
                    neuron['delta'] = 0
        ###################################################################
        kernelList = []
        errors = np.array([0] * len(net[0]['input']), dtype=float)
        for ind in range(noNet):
            for j, neuron in enumerate(net[ind]['input']):
                herror = 0
                for midneuron in (net[ind + noNet]['middle']):
                    herror += (midneuron['weights'][j]) * (midneuron['delta'])
                errors = np.append(errors, [herror])
            kernel = []
            for indexsub, neuron in enumerate(net[ind]['input']):
                neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
                if (np.isnan(neuron['delta'])):
                    neuron['delta'] = 0
                kernel.append(neuron['result'] - neuron['delta'])

            # kernel = np.reshape(kernel, (InNetInputNo[ind] // 2, InNetInputNo[ind] // 2))
            kernelList.append(kernel)
        return kernelList

    def updateWeights(self, net,output_net, input1, lrate, imSize, noNet):
        for ind in range(noNet):
            si = imSize[ind] // 2
            # inputs = np.reshape(input1[ind], (si * si, 1))
            inputs = [i[0] for i in (input1)]
            for neuron in (net[ind + noNet]['middle']):
                for j in range(len(inputs)):
                    neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= lrate * neuron['delta']

            for neuron in (output_net[0]['output']):
                for j, inputs in enumerate(net[ind + noNet]['middle']):
                    neuron['weights'][j] -= lrate * neuron['delta'] * inputs['result']
                neuron['weights'][-1] -= lrate * neuron['delta']

        return net

    def initKernel(self, kernelSizeList):

        KernelRankList1 = [np.random.uniform(low=-0.05, high=0.05, size=(i)) for i in kernelSizeList]
        return KernelRankList1

    def PNNChannels(self, noNet, output_net, epochs, train_fold_features, train_fold_labels, InNetInputNo, n_outputs,
                    labelvalue, lrate, hn, scale, kernelSizeList, imageSize,spearmanListsize):
        KernelRankList = self.initKernel(kernelSizeList)
        net = self.initialize_network(noNetwork, InNetInputNo, hnnolist, 10)
        z = len(train_fold_features)
        for epoch in range(epochs):
            rr = 1
            iterationoutput = np.array([])
            for i, row in enumerate(train_fold_features):
                xxx = np.array(list(row))
                trainfoldexpected = train_fold_labels[i]
                xxx1 = self.feedforwardByFilter(xxx, KernelRankList, kernelSizeList, imageSize, noNet,spearmanListsize)
                pooledList = self.maxpoolList(xxx1)
                outputs = self.forward_propagation(net,output_net, pooledList, n_outputs, labelvalue, scale, noNet)
                nnback = self.back_propagation(net,output_net, InNetInputNo, trainfoldexpected, outputs, n_outputs, labelvalue,
                                               scale, noNet)
                dPool2 = self.maxPoolBackwardList(nnback, xxx1, InNetInputNo,spearmanListsize)
                KernelRankList = self.backwardByFilter(dPool2, xxx, KernelRankList, kernelSizeList, InNetInputNo,
                                                       imageSize)
                net = self.updateWeights(net,output_net, pooledList, lrate, InNetInputNo, noNet)
                cc = self.calculateoutputTau([outputs, np.array(trainfoldexpected)])
                iterationoutput = np.append(iterationoutput, cc)
            rr = sum(iterationoutput) / z
            net = net

            print("-- Epoch %d", epoch)
            print("Tau per iteration ==> %.2f ", rr)
            # print("Kernel==>"+str(KernelRankList))
        net = net

        return net, outputs, KernelRankList

    def calculateoutputTau(self, iterationoutput):
        sum_Tau = 0
        tau, pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

    def backwardByFilter(self, DImage, originalDImage, KernelRankList, kernelsize, dSize, w):
        s = 1
        newKernelList = []
        for dx, ks in enumerate(kernelsize):
            f = kernelsize[dx]
            oneImage = DImage[dx]  # np.reshape(DImage[dx], ((w-f+1) * (h-f+1), 1))
            counter = 0
            KernelRank = KernelRankList[dx]
            curr_x = out_x = 0
            while curr_x + f <= w:
                xa = [a for a in originalDImage[curr_x:curr_x + f]]
                dkernel = self.DSpearmanImage(xa, oneImage[counter])
                for k in range(ks):
                    KernelRank[k] = KernelRank[k] - 0.07 * dkernel[k]
                curr_x += s
                counter += 1
                out_x += 1
            newKernelList.append(KernelRank)

        return newKernelList

    def feedforwardByFilter(self, oneDImage, KernelRanker, kernelsize, w, noNet, spearmanListsize):
        # start = time.time()
        s = 1
        spearmanLists=[]
        for dx, ks in enumerate(kernelsize):
            f = ks
            x_spearmansize=w - f + 1 
            kernelImage = KernelRanker[dx]
            spearmanList = [0]* x_spearmansize #, dtype=float)# np.zeros(shape = ( spearmansize, spearmansize ))
            curr_x = out_x = 0
            while curr_x + f <= w:
                xxx=oneDImage[ curr_x:curr_x + f]
                xx=self.calculate_rank([a.tolist() for a in xxx])
                corr, pv = spearmanr(xx, kernelImage)
                if np.isnan(corr):
                    corr = 0
                spearmanList[out_x]=corr
                curr_x += s
                out_x += 1
            # end = time.time()
            spearmanLists.append(spearmanList)
        return spearmanLists

    def CrossValidationAvg(self, output_net,X_train, y_train, InNetInputNo, noNet, hnnolist,
                           labelno, labelvalue, lrate, scale, epochs, bestvector, kernelSizeList, imageSize,spearmanListsize):
        avr_res = 0
        trainlabel =  [i for i in y_train]
        trainFeatures = X_train #[[self.imageAvg(i, imageSize)] for i in X_train]
        net, output, KernelRankList = self.PNNChannels(noNet, output_net, epochs, trainFeatures, trainlabel, InNetInputNo,
                                                       labelno, labelvalue, lrate, hnnolist, scale, kernelSizeList,
                                                       imageSize,spearmanListsize)
        return avr_res, net

    def calculate_rank(self, vector):
        a = {}
        rank = 1
        for num in sorted(vector):
            if num not in a:
                a[num] = rank
                rank = rank + 1
        return [a[i] for i in vector]

    def rescaleOneInstance(self, values, new_min, new_max):
        rowvalues = values
        old_min, old_max = min(rowvalues), max(rowvalues)
        outputf = [(new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min for v in rowvalues]
        return outputf

    def rankeImageVector(self, DimageList):
        imlist = []
        imlist = [self.calculate_rank(Dimage) for Dimage in (DimageList)]

        return imlist

    def training(self,output_net,thredno, epochs, X, y, InNetInputNo, noNetwork, labelno, labelvalue, lrate, hnnolist,
                 scale, kernelSizeList, imageSize,spearmanListsize):
        print("Thread "+str(thredno))
        lrlist = [lrate]
        scalelist = [scale]
        bestvector = [0, 0, 0, 0, 0]
        avresult = 0
        bestvresult = 0
        for lr1 in lrlist:
            for scl in scalelist:
                avresult, bestnet = self.CrossValidationAvg(output_net=output_net,X_train=X, y_train=y,
                                                                            InNetInputNo=InNetInputNo,
                                                                            noNet=noNetwork, hnnolist=hnnolist,
                                                                            labelno=labelno,
                                                                            labelvalue=labelvalue, lrate=lr1,
                                                                            scale=scale,
                                                                            epochs=epochs, bestvector=bestvector,
                                                                            kernelSizeList=kernelSizeList,
                                                                            imageSize=imageSize,spearmanListsize=spearmanListsize)
                print('crossv Prediction=%f , lr=%f', (avresult, lr1))
                if (avresult > bestvresult):
                    bestvresult = avresult
                    bestvector = [bestnet, lr1, bestvresult, scl]

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return bestnet

    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs

 
    def maxPoolBackwardList(self, dimageList, orig, w, spearmanListsize):
        # start = time.time()
        dPoolList = []
        for indx, dimage in enumerate(dimageList):
            dPool = self.maxpoolBackward(dimage, orig[indx], 2, 2, w, spearmanListsize[indx])
            dPoolList.append(dPool)
        # end = time.time()
        # print('maxPoolBackward %f',end - start)
        return dPoolList

    def maxpoolBackward(self, dpool, orig, f, s, w, spearmanListsize):
        '''
        Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
        '''
        orig_dim = len(orig)  # .shape)
        dout = orig
        curr_x = out_x = 0
        while curr_x + f <= orig_dim:
            b = np.nanargmax(orig[curr_x:curr_x + f])
            dout[curr_x + b] = dpool[out_x]
            curr_x += s
            out_x += 1

        return dout


    def maxpoolList(self, imageList):
        poolList = [self.maxpool(image) for image in imageList]
        return poolList

    def maxpool(self, image, f=2, s=2):
        '''
        Downsample `image` using kernel size `f` and stride `s`
        '''
        image = np.array(image)
        w_prev = len(image)
        w = int((w_prev - f) / s) + 1
        downsampled = [0] * w
        curr_x = out_x = 0
        while curr_x + f <= w_prev:
            xx = image[curr_x:curr_x + f]
            downsampled[out_x] = np.max(xx)
            curr_x += s
            out_x += 1
        downsampled1D = downsampled 
        return downsampled1D

    def predict(self, net, testFeatures, testlabel, labelno, labelvalue, scale, hnnolist, KernelRankList, imageSize,
                kernelSizeList, noNet,spearmanListsize):
        iterationoutput = np.array([])
        predictedList = []
        for i, row in enumerate((testFeatures)):
            xxx1 = np.array(list(row))
            testfoldlabels = list(testlabel[i])
            xxx1 = self.feedforwardByFilter(xxx1, KernelRankList, kernelSizeList, imageSize, noNet,spearmanListsize)
            pooled = self.maxpoolList(xxx1)  # maxpooling operation
            predicted = self.forward_propagation(net, pooled, labelno, labelvalue, scale, noNet)
            predictedList.append(predicted)
            iterationoutput = np.append(iterationoutput,
                                        [self.calculateoutputTau([predicted, np.array(testfoldlabels)])])

        avrre = sum(iterationoutput) / len(testFeatures)
        return avrre, predictedList

    def loadData1(self):
        sw1 = featuresize - kernelSizeList[0] + 1
        sw2 = featuresize- kernelSizeList[1] + 1
        sw3 = featuresize - kernelSizeList[2] + 1
        spearmanListsize = [sw1,sw2,sw3]
        InNetInputNo = [sw1 ,sw2 ,sw3]  
        
        # ========GNOM Sequence Imagery============================")
        from sklearn import preprocessing
        data = list()
        labels = list()
        print("==================================GNOM=============================")

        for clazz in range(1, 5):
            filename = "..//Data//LRData//GNOME.csv"
            gpsTrack = open(filename, "r")
            csvReader = csv.reader(gpsTrack)
            for row in csvReader:
                data.append(row[0:featuresize])
                labels.append(row[featuresize:featuresize + labelno])
            X = [[float(y) for y in x] for x in data]
            y = [[float(h) for h in l] for l in labels]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2,
                                                                                    random_state=1)
        trainList = X_train
        testimList = X_test

        trainY = y_train
        testY = y_test
        
        output_net = init_outputlayer(hnnolist, labelno)
        net1, tot_error2, predictedList = self.training(thredno=1,output_net=output_net,epochs=500,X=trainList, y=trainY
                                                        ,InNetInputNo=InNetInputNo,
                                                        noNetwork=noNetwork, labelno=labelno, labelvalue=labelvalue, lrate=lrate,
                                                        hnnolist=hnnolist,
                                                        scale=1, kernelSizeList=kernelSizeList, imageSize=featuresize,spearmanListsize=spearmanListsize)
        

        return net1

   
###############################################################

pnn = PNN()
train_error = pnn.loadData1()
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################