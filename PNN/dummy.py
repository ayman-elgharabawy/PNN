
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
from scipy.spatial.distance import cdist
from face_cropper import crop
from skimage import color
from skimage import io
from PIL import Image
from autocrop import Cropper
import glob
import cv2
import shutil 
import pickle
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
import PA
from PA import PAOneLayer
import collections
from keras.datasets import mnist
from keras.datasets import fashion_mnist


def PSS( xi, n, stepwidth=2):
        sum1 = 0
        b = 100
        for i in range(n):
            sum1 += -0.5 * (np.tanh(-b * (xi - (stepwidth * i))))
        sum1 = sum1 + (n / 2)
        return sum1

def rankeImageVector(DimageList):
    imlist = [calculate_rank(Dimage) for Dimage in (DimageList)]
    return imlist

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

def digit_sum(n):
    '''(int)->number
    Returns the sum of all the digits in the given integer, n'''
    if n<10:
        return n
    return n%10 + digit_sum(n//10)

def digital_root(n):
    if n < 10:
        return n
    return digital_root(digit_sum(n))


# (X1, y), (X_test, y_test) = fashion_mnist.load_data()
flatterd = []
# for tup in X1:
#     flatterd.append(tup.ravel())
# X = flatterd


# print("hi")
# plt.imshow('C:\\Github\\PNN\\PNN\\face1.png', cmap='gray')
# plt.show()
padded_face = cv2.imread('C:\\Github\\PNN\\PNN\\F2.jpg', cv2.IMREAD_GRAYSCALE)
for tup in padded_face:
    flatterd.append(tup.ravel())
X = flatterd
XX=np.reshape(X, (10000, 1))
XXX=[ i.tolist()[0] for i in XX]
dr=[]
for i in  XXX:
  dr.append((digital_root(i)))
print (dr)  

arr_2d = np.reshape(dr, (100, 100))
plt.imshow(arr_2d, cmap='gray')
plt.show()

# fig = plt.figure()
# ax = plt.axes()
# x = np.linspace(-5, 200, 1000)
# ax.plot(x,PSS(x,10,8))
# plt.axis([-2,100, -2, 12])
# plt.show()

