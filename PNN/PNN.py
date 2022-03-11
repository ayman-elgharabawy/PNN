# Copyright 2021 Ayman Elgharabawy. All Rights Reserved.
#     https://github.com/ayman-elgharabawy/PNN
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
# Use Python +3.7
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import zscore
from itertools import combinations, permutations
import csv
from random import randint
from scipy.stats.mstats import spearmanr
import scipy.stats as ss
import random
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import itertools
from sklearn.metrics import roc_curve, auc
import numpy.ma as ma
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime
import pandas as pd
from statistics import mean
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer



class PNN:

    def __init__(self):
        print("PNN Starting..")


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


    def DSpearman(self, output, expected):
        nn = len(expected)
        diflist=[]
        diflist = [2 * (output[i] - (expected[i])) for i in range(nn)]
        return diflist

    def calcConfusion(self, predictedList,y_test,ssteps):
            acc_sum = 0
            sens_sum = 0
            spec_sum = 0

            for i in range(len(y_test)):
                cc = self.calculate_rank(predictedList[i].tolist())
                ss = self.calculate_rank(y_test[i])
                cm1 = confusion_matrix(ss, cc,normalize='all')
                cm1=np.nan_to_num(cm1)
                # print('Confusion Matrix : \n', cm1)

                total1 = sum(sum(cm1))
                #####from confusion matrix calculate accuracy
                accuracy1 = (cm1[0, 0] + cm1[1, 1]) / (cm1[0, 0] + cm1[0, 1]+cm1[1, 0] + cm1[1, 1])
                acc_sum += accuracy1

                sensitivity1 =  cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                if(np.isnan(sensitivity1)):
                    sensitivity1=0
                sens_sum += sensitivity1

                specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                if(np.isnan(specificity1)):
                    specificity1=0
                spec_sum += specificity1

            print('Accuracy : ', acc_sum / len(y_test))
            print('Sensitivity : ', sens_sum / len(y_test))
            print('Specificity : ', spec_sum / len(y_test))


    def calculate_rank_predicted(self,vector,noranks):
        a = {}
        rank = noranks
        counter=0
        for num in sorted(vector, reverse=True):
            if num not in a:
                a[num] = rank
                counter+=1
                if ((rank)>counter):
                    rank-=1
        xx=[a[i] for i in vector]
        return xx

    def calculate_rank(self,vector):
        a = {}
        rank = 1
        for num in sorted(vector):
            if num not in a:
                a[num] = rank
                rank = rank + 1
        return [a[i] for i in vector]


    def drawROC(self,testY,probs,nolabels):
            plt.figure()
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            ##################################
            testY1, probs1=self.rocForIndex(probs,testY, 1,nolabels)
            for i in range(nolabels):
                fpr[i], tpr[i], _ = roc_curve(testY1[:, i], probs1[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["micro"], tpr["micro"], _ = roc_curve(testY1.ravel(), probs1.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]))
            for i in range(nolabels):
                plt.plot(fpr[i], tpr[i], label='ROC curve of Rank1 for label {0} (area = {1:0.2f})'
                                            ''.format(i, roc_auc[i]))
            #################################################
            ##################################
            # testY1, probs1=self.rocForIndex(probs,testY, 2,nolabels)
            # for i in range(nolabels):
            #     fpr[i], tpr[i], _ = roc_curve(testY1[:, i], probs1[:, i])
            #     roc_auc[i] = auc(fpr[i], tpr[i])
            # fpr["micro"], tpr["micro"], _ = roc_curve(testY1.ravel(), probs1.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            # plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'
            #             ''.format(roc_auc["micro"]))
            # for i in range(nolabels):
            #     plt.plot(fpr[i], tpr[i], label='ROC curve of Rank 2 for label {0} (area = {1:0.2f})'
            #                                 ''.format(i, roc_auc[i]))
            #################################################

            ##################################
            # testY1, probs1=self.rocForIndex(probs,testY, 3,nolabels)
            # for i in range(nolabels):
            #     fpr[i], tpr[i], _ = roc_curve(testY1[:, i], probs1[:, i])
            #     roc_auc[i] = auc(fpr[i], tpr[i])
            # fpr["micro"], tpr["micro"], _ = roc_curve(testY1.ravel(), probs1.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            # plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'
            #             ''.format(roc_auc["micro"]))
            # for i in range(nolabels):
            #     plt.plot(fpr[i], tpr[i], label='ROC curve of Rank 3 for label {0} (area = {1:0.2f})'
            #                                 ''.format(i, roc_auc[i]))
            #################################################
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()

    def rescale(self,values,new_min , new_max ): 
        totalrowoutput=[] 
        for rowvalues in (values):
            old_min, old_max = min(rowvalues), max(rowvalues)
            outputf = []
            for v in rowvalues:
                new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
                outputf.append(new_v)
            totalrowoutput.append(outputf)
        return totalrowoutput


    def rocForIndex(self,predictedx,expectedx,rnk,labelno):
            newexpected2=[]
            for index in range(len(expectedx)):
                newexpected1=[]
                for i in expectedx[index]:
                    if i==rnk:
                      newexpected1.append(1) 
                    else:
                      newexpected1.append(0)       
                newexpected2.append(newexpected1)

            newpredicted2=[]
            for index in range(len(predictedx)):
                newpredicted1=[]
                for i in predictedx[index]:
                    if rnk>=i:
                        number_dec = float(i/rnk)
                    else:
                        number_dec = float(rnk/i)
                    newpredicted1.append(number_dec)       
                newpredicted2.append(newpredicted1)
            return np.array(newexpected2) , np.array(newpredicted2)


    def generate_wt_drop(self,x, y,dropno):
        l =[]
        for i in range(x * y):
            D1=np.random.uniform(low=-0.005, high=0.005)
            l.append(D1)

        aa=(np.array(l).reshape(x, y))
        x = [randint(0, dropno) for p in range(0, y)]
        for ind ,i in enumerate(aa):
            if(ind in x):
              aa[ind]=[0]*y

        return aa , x

    def generate_wt(self,x, y):
        l =[]
        for i in range(x * y):
            l.append(np.random.uniform(low=-0.5, high=0.5))
        aa=(np.array(l).reshape(x, y))
        
        return aa

    def initialize_network(self,InNetInputNons, hiddenlist, outs,dropno):
        w1 = self.generate_wt(InNetInputNons, hiddenlist)
        w2 ,drop1= self.generate_wt_drop(hiddenlist, outs,dropno)
        return w1,w2 ,drop1

    


    def forward_propagation(self, w1,w2,drop1, input1, n_outputs, ssteps, scale, dropout):

        if dropout:
            for ind ,i in enumerate(w2):
                if(ind in drop1):
                  w2[ind]=[0]*16


        z1 = input1.dot(w1)# input from layer 1
        a1 = self.PSS(z1, ssteps, scale)# out put of layer 2


        z2 = a1.dot(w2)# input of out layer
        a2 = self.PSS(z2, ssteps, scale)# output of out layer

        return a2, drop1

    def back_propagation(self, w1,w2,drop1,lrate, input1,InNetInputNo, expected, outputs, n_outputs, ssteps, scale, cache,
                        dropout):
            if dropout:
                for ind ,i in enumerate(w2):
                    if(ind in drop1):
                       w2[ind]=[0]*16


            z1 = input1.dot(w1)# input from layer 1
            a1 = self.PSS(np.array([z1]), ssteps, scale)# out put of layer 2
            z2 = a1.dot(w2)# input of out layer
            a2 = self.PSS(z2, ssteps, scale)# output of out layer

            d2 = self.DSpearman(a2, np.array([expected]))
            xx=np.multiply(a1, 1-a1)
            ee=(w2.dot((np.array(d2).transpose()))).transpose()
            d1 = np.multiply(ee,(xx))

            w1_adj = np.array([input1]).transpose().dot(d1)
            w2_adj = a1.transpose().dot(d2)

            w1 = w1-(lrate*(w1_adj[0]))
            w2 = w2-(lrate*(w2_adj))

            return  w1,w2


    def PNNChannels( self, w1,w2,drop1, epochs, train_fold_features, train_fold_labels, InNetInputNo, n_outputs,
                    ssteps, lrate, hn, scale, dropout,dropno):
        z = len(train_fold_features)
        start1 = timer()
        cache = []
        for epoch in range(epochs):
            iterationoutput = np.array([])
            for i, row in enumerate((train_fold_features)):
                xxx = np.array(list(row))
                trainfoldexpected = list(train_fold_labels[i])
                outputs, drop1 = self.forward_propagation(w1,w2,drop1, xxx, n_outputs, ssteps, scale, dropout)
                w1,w2 = self.back_propagation(w1,w2,drop1,lrate,xxx, InNetInputNo, trainfoldexpected, outputs, n_outputs,
                                                    ssteps, scale, cache, dropout)
                cc = self.calculateoutputTau([outputs, np.array(trainfoldexpected)])
                iterationoutput = np.append(iterationoutput, cc)
            rr = sum(iterationoutput) / z
            w1,w2 = w1,w2

            print(' Epoch ' + str(epoch) + " rho=" + str(rr))
        w1,w2  = w1,w2 
        end1 = timer()
        print("Elapsed Training time")
        print(end1 - start1) # 
        return w1,w2 , outputs, cache


    def calculateoutputTau( self,iterationoutput):
        sum_Tau = 0
        tau, pv = spearmanr(iterationoutput[1], iterationoutput[0])
        sum_Tau += tau
        return sum_Tau

    def CrossValidationAvg(self, X_train, y_train,X_test,y_test, kfold, foldcounter, foldindex, InNetInputNo, hnnolist,
                        labelno, ssteps, lrate, scale, epochs, bestvector, useFold,
                        dropout,dropno):
        w1,w2,drop1 = self.initialize_network(InNetInputNo, hnnolist, labelno,dropno)

        avr_res = 0
        tot_etau = 0
        if (useFold):
            for idx_train, idx_test in kfold.split(X_train):
                foldindex += 1

                trainFeatures = [X_train[i] for i in idx_train]
                trainlabel = [y_train[i] for i in idx_train]
                
                testFeatures = [X_train[i]  for i in idx_test]
                testlabel = [y_train[i]  for i in idx_test]
                print("Fold Index="+str(foldindex))
                w1,w2 , output, cache = self.PNNChannels(w1,w2,drop1, epochs, trainFeatures, trainlabel,
                                        InNetInputNo,labelno, ssteps, lrate, hnnolist, scale,dropout,dropno)
                
                iterationoutput = self.predict(w1=w1,w2=w2 , testFeatures=testFeatures, testlabel=testlabel, labelno=labelno,
                                            ssteps=ssteps, scale=scale, hnnolist=hnnolist, dropout=dropout,dropno=dropno)

                tot_etau += iterationoutput[0]
               
                avr_res = tot_etau / foldcounter
                print(' Validation ' + str(avr_res) + " Fold index=" + str(foldindex))
        else:
            trainFeatures = [i for i in X_train]
            w1,w2, output, cache = self.PNNChannels(w1,w2,drop1, epochs, trainFeatures, y_train,
                                            InNetInputNo,labelno, ssteps, lrate, hnnolist, scale,dropout,dropno)

        print("###################################### Testing 20% of data ###########################") 
        start = timer()
        iterationoutput = self.predict(w1=w1,w2=w2 , testFeatures=X_test, testlabel=y_test, labelno=labelno,
                                            ssteps=ssteps, scale=scale, hnnolist=hnnolist, dropout=dropout,dropno=dropno) 
                                            # ...
        end = timer()
        print("Elapsed Testing time")
        print(end - start) # 
        self.calcConfusion(iterationoutput[1],y_test,ssteps)     
        # self.drawROC(y_test, np.array(iterationoutput[1]), 3)                                                                                     
        tot_etau = iterationoutput[0]
        avr_res = tot_etau
        return avr_res, w1,w2, cache

    def calculate_rank(self, vector):
        a = {}
        rank = 1
        for num in sorted(vector):
            if num not in a:
                a[num] = rank
                rank += 1
        return [a[i] for i in vector]

    def rescaleOneInstance(self, values, new_min, new_max):
        rowvalues = values
        old_min, old_max = min(rowvalues), max(rowvalues)
        outputf = [(new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min for v in rowvalues]
        return outputf

    def training(self, chunk, epochs, X, y, X_test,y_test,InNetInputNo, labelno, ssteps, lrate,
                hnnolist, scale, useFold, dropout,dropno, Fold):
        foldcounter = Fold
        kfold = sklearn.model_selection.KFold(foldcounter, shuffle=False)
        foldindex = 0
        lrlist = [lrate]
        scalelist = [scale]
        bestvector = [0, 0, 0, 0, 0]
        avresult = 0
        bestvresult = 0
        for lr1 in lrlist:
            for scl in scalelist:
                avresult, w1,w2, cache = self.CrossValidationAvg(kfold=kfold, foldcounter=foldcounter,
                                                                                foldindex=foldindex,
                                                                                X_train=X, y_train=y,
                                                                                X_test=X_test , y_test=y_test,
                                                                                InNetInputNo=InNetInputNo,
                                                                                hnnolist=hnnolist,
                                                                                labelno=labelno,
                                                                                ssteps=ssteps, lrate=lr1,
                                                                                scale=scale,
                                                                                epochs=epochs, bestvector=bestvector,
                                                                                dropout=dropout,
                                                                                useFold=useFold,
                                                                                dropno=dropno
                                                                                )
                print('Testing Data Prediction= ',avresult)
                if (avresult > bestvresult):
                    bestvresult = avresult
                    bestvector = [w1,w2, lr1, bestvresult, scl]
        # print('Best Vector = ',bestvector)
        return w1,w2, avresult

    def predict(self, w1,w2 , testFeatures, testlabel, labelno, ssteps, scale, hnnolist, dropout,dropoutno):
        
        iterationoutput = np.array([])
        predictedList = []
        cache=[]
        for i, row in enumerate((testFeatures)):
            xxx1 = np.array(list(row))
            testfoldlabels = list(testlabel[i])
            predicted, drop1,drop2 = self.forward_propagation(w1,w2, xxx1, labelno, ssteps, scale, dropout,dropoutno)
            predictedList.append(predicted)
            iterationoutput = np.append(iterationoutput,
                                        [self.calculateoutputTau([predicted, np.array(testfoldlabels)])])

        avrre = sum(iterationoutput) / len(testFeatures)
        return avrre, predictedList


    ###############################################################
    def isVarriant(list1, size):
        # insert the list to the set
        list_set = set(list1)
        # convert the set to the list
        unique_list = (list(list_set))
        if (len(unique_list) > size):
                return True
        else:
                return False


    def calculate_rank(self,vector):
        a = {}
        rank = 1
        for num in sorted(vector):
            if num not in a:
                a[num] = rank
                rank = rank + 1
        return [a[i] for i in vector]

    def loadData(self,filename, featuresno, labelno, ssteps,epochs,lrate,hn,Fold,useFold,scale,dropout,dropno):
        data = list()
        labels = list()
        alldata = list()
        scale=scale
        print("=================================="+filename+"=============================")
        filename1 =  filename
        gpsTrack = open(filename1, "r")
        csvReader = csv.reader(gpsTrack)
        next(csvReader)
        for row in csvReader :
                data.append(row[0:featuresno])
                labels.append(row[featuresno:featuresno + labelno])
                # alldata.append(row[:])

        y = np.array(labels)
        X = np.array(data)  

        train_features, test_features, train_labels, test_labels  =sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    
        X = X.astype(float)
        y = y.astype(float)

        train_features = train_features.astype(float)
        test_features = test_features.astype(float)

        train_labels = train_labels.astype(float)
        test_labels = test_labels.astype(float)

        train_features_norm=preprocessing.normalize(train_features)  #X
        test_features_norm=preprocessing.normalize(test_features)

        w1,w2,tot_error=self.training(chunk=1,epochs=epochs,ssteps=ssteps,Fold=Fold, X=train_features_norm,y=train_labels,
        X_test=test_features_norm , y_test=test_labels,InNetInputNo=featuresno,
        labelno=labelno,lrate=lrate,hnnolist=hn,scale=scale,dropout=dropout,dropno=dropno,useFold=useFold)
                            
        print('Done')
        return tot_error
    

###############################################################################################################################
###############################################################################################################################
##################################################################################################################################



