import numpy as np
# from IPython.display import Image,display
import matplotlib.pyplot as plt

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
# from scipy._lib.six import iteritems
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata
import networkx as nx
from sklearn import preprocessing
from numpy import transpose
from datetime import datetime
import mnist
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import test
from test import PAOneLayer1
import collections
from keras.datasets import mnist
from keras.datasets import fashion_mnist


def print_network(net):
    for i, layer in enumerate(net, 1):
        print("Layer {} ".format(i))
        for j, neuron in enumerate(layer, 1):
            print("neuron {} :".format(j), neuron)


###############################################################################################################################
def removeDataByLabelIndex(X, y, labelIndex):
    outputData = []
    outputLabels = []
    for indexl, row in enumerate(y):
        if row != labelIndex:
            outputData.append(X[indexl])
            outputLabels.append(row)
    return outputData, outputLabels


def removeDataByLabelList(X, y, labelList):
    outputData = []
    outputLabels = []
    for indexl, row in enumerate(y):
        for labelIndex in (labelList):
            if row == labelIndex:
                outputData.append(X[indexl])
                outputLabels.append(row)
    return outputData, outputLabels


def splitData(X_data, labels, classno):
    labels1 = []
    labels2 = []
    outputdata1 = []
    outputdata2 = []
    for index, i in enumerate(labels):
        if i > classno:
            labels1.append(i)
            outputdata1.append(X_data[index])
        else:
            labels2.append(i)
            outputdata2.append(X_data[index])

    return outputdata1, labels1, outputdata2, labels2


def trainTestingSplitter(train_features, train_labels):
    train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(train_features,
                                                                                                        train_labels,
                                                                                                        test_size=0.3,
                                                                                                        random_state=1)
    X = np.array([list(item) for item in train_features])
    y = train_labels
    X1 = np.array([list(item) for item in test_features])
    y1 = test_labels
    return X, y, X1, y1


def SortImages(X, y):
    newFacesList = []
    newLabelsList = []
    newFacesList.append(X[0])
    newLabelsList.append(y[0])
    distList = []
    for i in range(len(X)):
        closetDist = 100
        closetIndex = 0
        for j in range(i + 1, len(X)):
            dist1 = SW(X[i]) - SW(X[j])
            if (closetDist > dist1):
                closetDist = dist1
                closetIndex = j
        newFacesList.append(X[closetIndex])
        newLabelsList.append(y[closetIndex])
        distList.append(closetDist)
    return newFacesList, newLabelsList, distList


def SW(X):
    # S_W=0
    S_W = []  # np.zeros((128,128))
    meanv = getMeanVector(X)
    for mv in meanv:
        class_sc_mat = 0  # scatter matrix for every class
        class_sc_mat += (X - mv).dot((X - mv).T)
        S_W.append(class_sc_mat)  # sum class scatter matrices
    # print('within-class Scatter Matrix:\n', S_W)
    return S_W[0]


def getMeanVector(X):
    np.set_printoptions(precision=4)
    mean_vectors = []
    mean_vectors.append(np.mean(X, axis=0))
    # print('Mean Vector class  %s\n' %( mean_vectors))
    return mean_vectors


def labelBinarySplitter(labels, splitpoint):
    newlist = []
    for lab in labels:
        if (lab > splitpoint):
            newlist.append([2, 1])
        else:
            newlist.append([1, 2])
    return newlist


def categorDataByBinaryResult(originalData, y, trainedlabels):
    newlabel1 = []
    newlabel2 = []

    newdata1 = []
    newdata2 = []

    for index, lab in enumerate(originalData):
        if (round(trainedlabels[index]) == 2):
            newlabel1.append(y[index])
            newdata1.append(originalData[index])
        else:
            newlabel2.append(y[index])
            newdata2.append(originalData[index])

    return newdata1, newlabel1, newdata2, newlabel2


def convertMultilabel(train_labels, labelno):
    newLabels = np.array([[0] * labelno] * len(train_labels))

    for index, row in enumerate(train_labels):
        newLabels[index, row] = 1

    return newLabels


def calculate_rank(vector):
    a = {}
    rank = 1
    for num in sorted(vector):
        if num not in a:
            a[num] = rank
            rank = rank + 1
    return [a[i] + 1 for i in vector]


def horizontalrankImagePixel(DimageList):
    imlist = []
    for Dimage in (DimageList):
        newIm = []
        for row in (Dimage):
            # newRow = calculate_rank(row)
            newRow = ss.rankdata(row)
            newIm.append(np.array(newRow, dtype=np.uint8))
        imlist.append(newIm)
    return imlist


def binaryLabels(labels):
    newlist = []
    for lab in labels:
        if (lab == 0):
            newlist.append([1, 2])
        else:
            newlist.append([2, 1])
    return newlist


def convert1Dto2D(arr):
    imlist = []
    for ima in arr:
        arr_2d = np.reshape(ima, (28, 28))
        imlist.append(arr_2d)
    return imlist


def convert2Dto1D(rankedImages):
    imlist = []
    for ima in rankedImages:
        flatImage = list(np.concatenate(ima).flat)
        imlist.append(flatImage)

    return imlist


def rankeImageVector(DimageList):
    imlist = [calculate_rank(Dimage) for Dimage in (DimageList)]
    return imlist


def getUniqueDataValue(inputdata):
    # compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    uniqueList = [[0] * len(inputdata[0])]

    counter = len(inputdata)
    for i in range(len(inputdata)):
        for j in range(i + 1, len(inputdata)):
            if (inputdata[i].tolist() == inputdata[j].tolist()):
                counter -= 1

    return counter


def loadData():
    print("==================================Mnist Dataset=============================")
    (X, y), (X_test, y_test) = fashion_mnist.load_data()
    flatterd = []
    for tup in X:
        flatterd.append(tup.ravel())
    X = flatterd
    flatterd = []
    for tup in X_test:
        flatterd.append(tup.ravel())
    X_test = flatterd

    return X, X_test, y, y_test


X, X_test, y, y_test = loadData()
cutval = 10  # len(X)
X = X[0:cutval]
X_test = X_test[0:cutval]
y = y[0:cutval]
y_test = y_test[0:cutval]
###################################################################################
pnnEncoderDecoder = PAOneLayer1()
# X123, y123 = removeDataByLabelList(X, y, [0])


X123 = rankeImageVector(X)
aaa=np.amax(X123)
# uniqueNumber=getUniqueDataValue(X)
# print(uniqueNumber)
net1 = pnnEncoderDecoder.loadData(X=X123, y=X123, featuresno=784, labelno=784, iteration=2500,
                                  lrate=0.007, middle=1, scale=5, ssteps=aaa)
# print_network(net1)