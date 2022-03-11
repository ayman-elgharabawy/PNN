import numpy as np
from sympy import *
import pandas as pd 


def getrank(stringval):
    rankmap = {}
    ranklist=list()
    templist=list()
    keyindexlist=list()
    counter=1
    for i in range(len(stringval)):
        if(stringval[i]!='<' and stringval[i]!='>'):
         xx=stringval[i]
         templist.append(xx)
        else:
         xxx=','.join(templist) 
         rankmap[xxx]=counter
         templist[:] = []
         counter=counter+1
    xxx=','.join(templist) 
    rankmap[xxx]=counter
    templist[:] = []

    labels=[]
    for c in ['a','b','c','d','e','f','g']:
        for i in range(len(rankmap)):
           ll= list(rankmap)[i]
           if c in ll:
               labels.append(i+1)

    return labels


def labelsprocessing(labelsarray):
    ranklist=list()
    for i in  labelsarray:
       row1=getrank(i)
       ranklist.append(row1)    
    return ranklist



def loadDatapermutation(filename):
    filename ='C:\\Ayman\PhDThesis\\' + filename + '.txt'
    file_handler = open(filename, "r")  
    data = pd.read_csv(file_handler, sep = ",") 
    file_handler.close() 

    #############algea###############################
    V1 = {'winter': 1,'spring': 2,'autumn': 3,'summer': 4} 
    V2 = {'small_': 1,'medium': 2,'large_': 3} 
    V3 = {'high__':1,'medium':2,'low___':3}

    data.V1 = [V1[item] for item in data.V1] 
    data.V2 = [V2[item] for item in data.V2]  
    data.V3 = [V3[item] for item in data.V3] 
  

    alldata= pd.concat([data.V1,data.V2,data.V3,data.V4,data.V5,data.V6,data.V7,data.V8,data.V9,data.V10,data.V11],1)
    labelsarray= data.ranking
    labels=labelsprocessing(labelsarray)
    dd=  alldata.values.tolist()
    myarray1 = np.array(dd)
    myarray2 = np.array(labels)
    newarray = np.concatenate((myarray1, myarray2),axis=1)
    ALL = newarray.tolist()

    fmt =  '%1.1f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%1d', '%1d', '%1d', '%1d', '%1d', '%1d', '%1d'
    np.savetxt('C:\\Ayman\PhDThesis\\algae_ranked.txt',ALL,fmt=fmt)


loadDatapermutation('algae')