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
import time
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
from keras.datasets import mnist

from scipy.stats.mstats import spearmanr
from keras.datasets import cifar10
import pandas as pd
from keras.utils import np_utils
from extra_keras_datasets import emnist
from keras.datasets import fashion_mnist
from statistics import mean
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import to_categorical
from sklearn import metrics

# import tfds.DatasetBuilder.as_dataset

class PNN:

    def __init__(self):
        print("PNN Starting..")

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
        diflist2 = [(2 * (rhs-i)) for i in output]
        return np.array(diflist2)

    def DSpearman(self, output, expected):
        n = len(expected)
        diflist = [2 * (output[i] - int(expected[i])) for i in range(n)]
        return diflist

    def print_network(self, net):  # ,epoch,tau,row1):
        # with open('C:\\ayman\\PhDThesis\\log\\SGPN_output.txt', 'a') as f:
        # print("------------------------------------------------------ Epoch "+str(epoch)+" ---------------------------------------\n")
        # print("Input row:" +str(row1))
        for i, layer in enumerate(net, 1):
            if (i == 1):
                print("=============== Middle layer =============== \n")
                for neuron in (net[0]['middle']):
                    print("Weights  :" + str(neuron['weights']))
                    print("delta  :" + str(neuron['delta']))
                    print("result  :" + str(neuron['result']))
            else:
                print("=============== Output layer =============== \n")
                for neuron in (net[1]['output']):
                    print("Weights  :" + str(neuron['weights']))
                    print("delta  :" + str(neuron['delta']))
                    print("result  :" + str(neuron['result']))
        # print("==== Roh Correlation = "+str(tau)+"======\n")

    def initialize_network(self, iInNet, InNetInputNons, hiddenlist, outs):
        output_neurons = outs
        net = list()
        sumhn = sum(hiddenlist)

        for inp in range(iInNet):
            input_layer = {'input': [{'result': 0} for i in range((InNetInputNons[inp] // 2) ** 2)]}
            net.append(input_layer)

        for inp in range(iInNet):
            hidden_layer = {
                'middle': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=(InNetInputNons[inp] // 2) ** 2)}
                           for i in range(hiddenlist[inp])]}
            net.append(hidden_layer)

        output_layer = {
            'output': [{'weights': np.random.uniform(low=-0.05, high=0.05, size=sumhn)} for i in range(output_neurons)]}
        net.append(output_layer)

        return net

    def forward_propagation(self, net, input1, n_middle, n_outputs, labelvalue, scale,noNet):
        start = time.time()
        for ind, row1 in enumerate(input1):
            for indx, neuron in enumerate(net[ind]['input']):
                neuron['result'] = row1[indx]

        prev_input=np.array([0]* n_middle[0] , dtype=float)
        for inn,neuron in enumerate(net[1]['middle']):
            sum1 = neuron['weights'].T.dot(row1)
            result = self.SSS(sum1, labelvalue, scale)
            neuron['result'] = result
            prev_input[inn]=result
        row1 = prev_input

        prev_input=np.array([0]* n_outputs , dtype=float)
        for idx,neuron in enumerate(net[2]['output']):
            sum1 = neuron['weights'].T.dot(row1)
            result = self.SSS(sum1, labelvalue, scale)
            if np.isnan(result):
                result = 0
            neuron['result'] = result
            prev_input[idx]=result

        row1 = prev_input
        end = time.time()
        # print('forwardpropagation %f',end - start) 
        return row1

    def back_propagation(self, net, InNetInputNo, expected, outputs,hnlist, n_outputs, labelvalue, scale):
        start = time.time()
        results = list()
        errors=np.array([0]* n_outputs , dtype=float)
        results = [neuron['result'] for neuron in net[2]['output']]
        errors = self.DSpearman(results, expected)

        for indexsub, neuron in enumerate(net[2]['output']):
            neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
            if (np.isnan(neuron['delta'])):
                neuron['delta'] = 0
        errors=np.array([0]* hnlist[0] , dtype=float)        
        for ind in range(1):
            for j, neuron in enumerate(net[1]['middle']):
                herror = 0
                for outneuron in (net[2]['output']):
                    herror += (outneuron['weights'][j]) * (outneuron['delta'])
                errors[j] = herror

            for indexsub, neuron in enumerate(net[1]['middle']):
                neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
                if (np.isnan(neuron['delta'])):
                    neuron['delta'] = 0
        #################################input##################################
        kernelList = []
        errors=np.array([0]* len(net[0]['input']) , dtype=float) 
        for ind in range(1):
            for j, neuron in enumerate(net[ind]['input']):
                herror = 0
                for inn,midneuron in enumerate(net[1]['middle']):
                    herror += (midneuron['weights'][j]) * (midneuron['delta']) 
                errors[inn]= herror
            ksize=InNetInputNo[ind] // 2
            kernel=np.array([0]* ksize*ksize , dtype=float)

            for indexsub, neuron in enumerate(net[ind]['input']):
                neuron['delta'] = errors[indexsub] * self.dSSS(neuron['result'], labelvalue, scale)
                if (np.isnan(neuron['delta'])):
                    neuron['delta'] = 0
                kernel[indexsub]=neuron['result'] - neuron['delta']

            kernel = np.reshape(kernel, (ksize, ksize))
            kernelList.append(kernel)
        end = time.time()
        # print('backpropagation %f',end - start)     
        return kernelList

    def updateWeights(self, net, input1, lrate, imSize, noNet):
        start = time.time()
        si = imSize[0] // 2
        nn=si * si
        inputs = np.reshape(input1[0], (si * si, 1))
        inputs = [i[0] for i in (inputs)]
        for neuron in (net[1]['middle']):
            for j in range(nn):
                neuron['weights'][j] -= lrate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= lrate * neuron['delta']
        for neuron in (net[2 * noNet]['output']):
            for j, inputs in enumerate(net[1]['middle']):
                neuron['weights'][j] -= lrate * neuron['delta'] * inputs['result']
            neuron['weights'][-1] -= lrate * neuron['delta']
        end = time.time()
        # print('updateWeights %f',end - start) 
        return net

    def initKernel(self, kernelSizeList):

        KernelRankList1 = [np.random.uniform(low=-0.05, high=0.05, size=(i * i)) for i in kernelSizeList]
        return KernelRankList1

    def PNNChannels(self, noNet, net, epochs, train_fold_features, train_fold_labels, InNetInputNo, n_outputs,
                    labelvalue, lrate, hn, scale, kernelSizeList, imageSize):
        KernelRankList = self.initKernel(kernelSizeList)
        z = len(train_fold_features)
        for epoch in range(epochs):
            rr = 0
            iterationoutput = []
            for i, row in enumerate((train_fold_features)):
                xxx = np.array(list(row))
                trainfoldexpected = list(train_fold_labels[i])
                xxx1 = self.feedforwardByFilter(xxx, KernelRankList, kernelSizeList, imageSize,noNet)
                pooledList = self.maxpoolList(xxx1)
                outputs = self.forward_propagation(net, pooledList,hn, n_outputs, labelvalue, scale, noNet)
                nnback = self.back_propagation(net, InNetInputNo, trainfoldexpected, outputs,hn, n_outputs, labelvalue,
                                               scale)
                dPool2 = self.maxPoolBackwardList(nnback, xxx1, InNetInputNo)
                KernelRankList = self.backwardByFilter(dPool2, xxx, KernelRankList, kernelSizeList, InNetInputNo,
                                                       imageSize)
                net = self.updateWeights(net, pooledList, lrate, InNetInputNo, noNet)
                cc = self.calculateoutputTau([outputs, np.array(trainfoldexpected)])
                iterationoutput = np.append(iterationoutput, cc)
            rr = sum(iterationoutput) / z
            net = net

            print('-- Epoch %d'% epoch)
            print('Tau per iteration ==> %f' % rr)
            # print("Kernel==>"+str(KernelRankList))
        net = net

        return net, outputs, KernelRankList

    def calculateoutputTau(self, iterationoutput):
        sum_Tau = 0
        tau, pv = ss.spearmanr(iterationoutput[1], iterationoutput[0])
        if not np.isnan(tau):
            sum_Tau += tau
        return sum_Tau

    def backwardByFilter(self, DImage, originalDImage, KernelRankList, kernelsize, dSize, originalImagesize):
        start = time.time()
        s = 1
        newKernelList = []
        for dx, ks in enumerate(kernelsize):
            f = kernelsize[dx]
            oneImage = np.reshape(DImage[dx], (dSize[dx] * dSize[dx], 1))
            twoDImage = np.reshape(originalDImage, (originalImagesize, originalImagesize))
            curr_y = out_y = 0
            counter = 0
            KernelRank = KernelRankList[dx]
            while curr_y + f <= originalImagesize:
                curr_x = out_x = 0
                while curr_x + f <= originalImagesize:
                    dkernel = self.DSpearmanImage([a.tolist()[0] for a in
                                                   np.reshape(twoDImage[curr_y:curr_y + f, curr_x:curr_x + f],
                                                              (ks * ks, 1))], oneImage[counter])
                    for k in range(ks * ks):
                        KernelRank[k] = KernelRank[k] - 0.07 * dkernel[k]
                    curr_x += s
                    counter += 1
                    out_x += 1
                curr_y += s
                out_y += 1
            newKernelList.append(KernelRank)
        end = time.time()
        # print('backwardByFilter %f',end - start)      
        return newKernelList

    def feedforwardByFilter(self, oneDImage, KernelRanker, kernelsize, imagesize, noNet):
        start = time.time()
        s = 1
        curr_y = out_y = 0
        f = kernelsize[0]
        spearmansize=imagesize - f + 1 
        twoDImage =  np.reshape(oneDImage, (imagesize, imagesize))
        kernelImage = KernelRanker[0]
        spearmanList = np.array([[0]* spearmansize]*spearmansize , dtype=float)# np.zeros(shape = ( spearmansize, spearmansize ))
        while curr_y + f <= imagesize:
            curr_x = out_x = 0
            while curr_x + f <= imagesize:
                corr, pv = spearmanr(self.calculate_rank([a.tolist()[0] for a in np.reshape(
                    twoDImage[curr_y:curr_y + f, curr_x:curr_x + f], (f* f, 1))]), kernelImage)
                if np.isnan(corr):
                    corr = 0
                spearmanList[out_x,out_y]=corr
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        end = time.time()
        # print('feedforwardByFilter %f',end - start)    
        return [spearmanList]
    
    def decodeLabelValues(self,binaryMatrix,expectedMatrix):
        decoded=[]
        for indx1, label in enumerate(binaryMatrix):
            expectedrow=expectedMatrix[indx1]
            for index ,j in enumerate(expectedrow):
              if(j==1):
                decoded.append(label[index])
                break   

        return  decoded  

    def decodeLabels(self,bindayMatrix):
        decoded=[]
        for label in enumerate(bindayMatrix):
            for index ,j in enumerate(label[1].tolist()):
              if(j==1):
                decoded.append(index+1)
                break   

        return  decoded   

    def plot_roc_curve(self,fpr, tpr):
        plt.plot(fpr, tpr, color='orange', label='Preference Net')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def CrossValidationAvg(self, kfold, foldcounter, foldindex, X_train, y_train, InNetInputNo, noNet, hnnolist,
                           labelno, labelvalue, lrate, scale, epochs, bestvector, kernelSizeList, imageSize):
        net = self.initialize_network(noNet, InNetInputNo, hnnolist, labelno)
        avr_res = 0
        tot_etau = 0
        for idx_train, idx_test in kfold.split(X_train):
            foldindex += 1
            trainFeatures = np.array([0]* 55201 , dtype=float) 
            testFeatures =np.array([0]* 4799 , dtype=float) 
            trainlabel =np.array([0]* 55201 , dtype=float)  
            testlabel =np.array([0]* 4799 , dtype=float)  

            trainlabel = [y_train[i] for i in idx_train]
            trainFeatures = [[self.imageAvg(X_train[i], imageSize)] for i in idx_train]

            testlabel = [y_train[i] for i in idx_test]
            testFeatures = [[self.imageAvg(X_train[i], imageSize)] for i in idx_test]

            net, output, KernelRankList = self.PNNChannels(noNet, net, epochs, trainFeatures, trainlabel, InNetInputNo,
                                                           labelno, labelvalue, lrate, hnnolist, scale, kernelSizeList,
                                                           imageSize)

            # self.print_network(net)
            iterationoutput = self.predict(net=net, testFeatures=testFeatures, testlabel=testlabel, labelno=labelno,
                                           labelvalue=labelvalue, scale=scale, hnnolist=hnnolist,
                                           KernelRankList=KernelRankList, kernelSizeList=kernelSizeList,
                                           imageSize=imageSize, noNet=noNet)
        
        avr_res = tot_etau / foldcounter
        print('Final average %.2f Folds test Result %.8f'% (foldcounter, avr_res))

        return avr_res, net , KernelRankList 

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
        # imlist =np.array([])
        imlist=[ self.calculate_rank(Dimage) for Dimage in (DimageList)]
        return imlist

    def training(self, epochs, X, y, X_test, Y_test, InNetInputNo, noNetwork, labelno, labelvalue, lrate, hnnolist,
                 scale, kernelSizeList, imageSize):
        foldcounter = 10
        kfold = sklearn.model_selection.KFold(foldcounter, shuffle=True, random_state=1)
        foldindex = 0
        lrlist = [lrate]  # ,0.09,0.1,0.2,0.3,0.4,0.5]
        scalelist = [scale]
        bestvector = [0, 0, 0, 0, 0]
        avresult = 0
        bestvresult = 0
        for lr1 in lrlist:
            for scl in scalelist:
                avresult, bestnet, KernelRankList = self.CrossValidationAvg(kfold=kfold, foldcounter=foldcounter,
                                                                           foldindex=foldindex,
                                                                            X_train=X, y_train=y,
                                                                            InNetInputNo=InNetInputNo,
                                                                            noNet=noNetwork, hnnolist=hnnolist,
                                                                            labelno=labelno,
                                                                            labelvalue=labelvalue, lrate=lr1,
                                                                            scale=scale,
                                                                            epochs=epochs, bestvector=bestvector,
                                                                            kernelSizeList=kernelSizeList,
                                                                            imageSize=imageSize)
                print('crossv Prediction=%f , lr=%f'% (avresult, lr1))
                if (avresult > bestvresult):
                    bestvresult = avresult
                    bestvector = [bestnet, lr1, bestvresult, scl]
        testFeatures=[self.rescaleOneInstance(self.imageAvg(X_test[i], imageSize), -1, 1) for i, ii in enumerate(X_test)]    
        iterationoutput, predictedList = self.predict(net=bestnet, testFeatures=testFeatures, testlabel=Y_test,
                                                      labelno=labelno, labelvalue=labelvalue, scale=scale,
                                                      hnnolist=hnnolist, KernelRankList=KernelRankList,
                                                      kernelSizeList=kernelSizeList, imageSize=imageSize,
                                                      noNet=noNetwork)
        
        probs=self.decodeLabelValues(predictedList,Y_test)
        decodedPredictedd=self.decodeLabels(Y_test)
        fpr, tpr, thresholds = metrics.roc_curve(decodedPredictedd,  probs, pos_label=2)
        print(fpr)
        print(tpr)
        self. plot_roc_curve(tpr,fpr)

        print(">>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return bestnet, avresult, predictedList

    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs

    def maxPoolBackwardList(self, dimageList, orig, InNetInputNo):
        start = time.time()
        dPoolList = []
        for indx, dimage in enumerate(dimageList):
            dPool = self.maxpoolBackward(dimage, orig[indx], 2, 2, InNetInputNo[indx])
            dPoolList.append(dPool)
        end = time.time()
        # print('maxPoolBackward %f',end - start)      
        return dPoolList


    def maxpoolBackward(self, dpool, orig, f, s, imageSize):
        '''
        Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
        '''
        orig = np.reshape(orig, (imageSize, imageSize))
        (orig_dim, _) = orig.shape
        dout =  np.zeros(orig.shape)
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                (a, b) = self.nanargmax(orig[curr_y:curr_y + f, curr_x:curr_x + f])
                dout[curr_y + a, curr_x + b] = dpool[out_y, out_x]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

        return dout

    def imageAvg(self, oneDImage, imageSize):
        start = time.time()
        image = np.reshape(oneDImage, (imageSize, imageSize))
        s , f = 1 ,  2
        avgImage = np.array([[0] * imageSize] * imageSize, dtype=float)
        curr_y = out_y = 0
        while curr_y + f <= imageSize:
            curr_x = out_x = 0
            while curr_x + f <= imageSize:
                avgImage[out_y, out_x] = mean(
                    [a.tolist()[0] for a in np.reshape(image[curr_y:curr_y + f, curr_x:curr_x + f], (4, 1))])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

        avgImage1D = np.reshape(avgImage, (imageSize * imageSize, 1))
        avgImage1D = [i.tolist()[0] for i in avgImage1D]
        end = time.time()
        # print('Image Average %f',end - start)  
        return avgImage1D

    def maxpoolList(self, imageList):
        start = time.time()
        poolList = [self.maxpool(image) for image in imageList]
        end = time.time()
        # print('maxpool %f',end - start)  
        return poolList

    def maxpool(self, image, f=2, s=2):
        '''
        Downsample `image` using kernel size `f` and stride `s`
        '''
        image = np.array(image)
        h_prev, w_prev = image.shape
        h = int((h_prev - f) / s) + 1
        w = int((w_prev - f) / s) + 1
        downsampled = np.zeros((h, w))
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                xx = image[curr_y:curr_y + f, curr_x:curr_x + f]
                downsampled[out_y, out_x] = np.max(xx)
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

        downsampled1D = [i.tolist()[0] for i in np.reshape(downsampled, (w * h, 1))]
        return downsampled1D

    def predict(self, net, testFeatures, testlabel, labelno, labelvalue, scale, hnnolist, KernelRankList, imageSize,
                kernelSizeList, noNet):
        nn=len(testFeatures)
        iterationoutput = np.array([0]*nn,dtype=float)
        predictedList = []
        for i, row in enumerate((testFeatures)):
            xxx1 = np.array(list(row))
            testfoldlabels = list(testlabel[i])
            xxx1 = self.feedforwardByFilter(xxx1, KernelRankList, kernelSizeList, imageSize, noNet)
            pooled = self.maxpoolList(xxx1)  # maxpooling operation
            predicted = self.forward_propagation(net, pooled,hnnolist, labelno, labelvalue, scale, noNet)
            predictedList.append(predicted)
            iterationoutput = np.append(iterationoutput,
                                        [self.calculateoutputTau([predicted, np.array(testfoldlabels)])])

        avrre = sum(iterationoutput) / len(testFeatures)
        return avrre, predictedList

    def loadData1(self):
        imageSize = 28
        noNetwork = 1
        kernelSizeList = [5]#, 10, 20]
        InNetInputNo = [(imageSize - 5 + 1)]#, imageSize - 10 + 1 ** 2, imageSize - 20 + 1 ** 2]
        hnnolist = [300]#, 1000, 400]
        print("==================================Hand Writing=============================")
        from sklearn import preprocessing
        (trainX, trainY), (testX, testY) = mnist.load_data()

        trainList = trainX[0:100] # [tf.image.rgb_to_grayscale(im).numpy() for im in trainX]
        testimList = testX[0:100] #[tf.image.rgb_to_grayscale(im).numpy() for im in testX]

        trainY = to_categorical(trainY[0:100])
        testY = to_categorical(testY[0:100])
        net1, tot_error2, predictedList = self.training(epochs=3, X=trainList, y=trainY, X_test=testimList,
                                                        Y_test=testY, InNetInputNo=InNetInputNo,
                                                        noNetwork=noNetwork, labelno=10, labelvalue=1, lrate=0.07,
                                                        hnnolist=hnnolist,
                                                        scale=1, kernelSizeList=kernelSizeList, imageSize=imageSize)
        return net1


###############################################################
pnn = PNN()
train_error = pnn.loadData1()
###############################################################################################################################
###############################################################################################################################
##################################################################################################################################