import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# from IPython.display import Image,display
import matplotlib.pyplot as plt
data = []
labels = []
alldata = []

# XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
# X=XORdata[:,0:2]
# y=XORdata[:,-1]


def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)


def rocForIndex(predictedx,expectedx,rnk,labelno):

            newexpected2=[]
            for index in range(len(expectedx)):
                newexpected1=[]
                for id,i in enumerate(expectedx[index].tolist()):
                    if i==1:
                      newexpected1.append(1) 
                    else:
                      newexpected1.append(0)       
                newexpected2.append(newexpected1)

            newpredicted2=[]
            for index in range(len(predictedx)):
                newpredicted1=[]
                for id,i in enumerate(predictedx[index]):
                    if i==1:
                        newpredicted1.append(1)
                    else:
                        newpredicted1.append(0)
                           
                newpredicted2.append(newpredicted1)
            return np.array(newexpected2) , np.array(newpredicted2)


def decodeBinaryToBinaryPred(expectedrows):

    declist_expected=[]
    for row in expectedrows:
        expectedx=np.argmax(row[0])+1
        if(expectedx==1):
            expectedx=[1,0,0]
        if(expectedx==2):
            expectedx=[0,1,0]
        if(expectedx==3):
            expectedx=[0,0,1]
        declist_expected.append(expectedx)    
    return declist_expected 

def decodeBinaryToInt(expectedrows):
    declist_expected=[]
    for row in expectedrows:
        expectedx=np.argmax(row)+1
        declist_expected.append(expectedx)    

    return declist_expected 


def drawROC(testY,probs,nolabels):
        # probs=[i.tolist()[0] for i in probs]
        plt.figure()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        ##################################
        testY1= decodeBinaryToInt(testY)
        probs1= decodeBinaryToBinaryPred(probs)

        testY2, probs2=rocForIndex(probs1,testY, 1,3)
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(testY2[:, i], probs2[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(testY2.ravel(), probs2.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'
                        ''.format(roc_auc["micro"]))
        for i in range(3):
                plt.plot(fpr[i], tpr[i], label='ROC curve of Rank1 for label {0} (area = {1:0.2f})'
                                            ''.format(i, roc_auc[i]))
        #################################################
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

def initialize_network():
    
    input_neurons=len(X[0])
    hidden_neurons=input_neurons+1
    output_neurons=3
    
    n_hidden_layers=1
    
    net=list()
    
    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])
            
        hidden_layer = [ { 'weights': np.random.uniform(low=-0.5, high=0.5,size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)
    
    output_layer = [ { 'weights': np.random.uniform(low=-0.5, high=0.5,size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)
    
    return net



def  activate_sigmoidactivat (sum):
    return (1/(1+np.exp(-sum)))


def  forward_propagationforward (net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)
            
            result=activate_sigmoid(sum)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input
    
    return row



def sigmoidDerivative(output):
    return output*(1.0-output)


def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results) 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
     return net

def updateWeights(net,input,lrate):
    
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
    return net

def  training(net, epochs,lrate,n_outputs):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs,net=forward_propagation(net,row)
            
            expected=[0.0 for i in range(n_outputs)]
    
            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
            net=back_propagation(net,row,expected)
            net=updateWeights(net,row,0.05)
        if epoch%10 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors ,net


# Make a prediction with a network# Make a 
def predict(network, rows):
    totalvals=[]
    for row in rows:
        outputs = forward_propagation(network, row)
        totalvals.append(outputs)

    return totalvals

def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)
            
            result=activate_sigmoid(sum)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input
    
    return row ,net


def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def numericlabels(data1):

    integer_list=( [list( map(int,i) ) for i in data1] )
    return integer_list

def numericdataandllabels(data):
    numericdatavalues = list()
    temprownumeric = list()
    for i in range(0, (len(data))):
        row = data[i]
        temprow = list()
        if(len(row) == 0):
            del data[i]
            continue
        for j in range(len(row)-3, len(row)):
            temp = row[j]
            row[j] = temp[1:]
        floatvalues = [float(item) for item in row]
        temprownumeric.append(floatvalues)
    return temprownumeric

def calcConfusion( predictedList,y_test_original):
        acc_sum = 0
        sens_sum = 0
        spec_sum = 0

        # for i in range(len(y_test_original)):
        for i in range(len(y_test_original)):
            ss = y_test_original# self.calculate_rank(y_test[i])
            a=ss.tolist()[i]
            b=predictedList[i]
            bb=np.argmax(b[0])+1
            if(bb==1):
              aa=[1,0,0]
            if(bb==2):
              aa=[0,1,0]
            if(bb==3):
              aa=[0,0,1]  
            cm1 = confusion_matrix(a, aa,normalize='all')
            # print('Confusion Matrix : \n', cm1)
            cm1=np.nan_to_num(cm1)
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

        print('Accuracy : ', acc_sum / len(y_test_original))
        print('Sensitivity : ', sens_sum / len(y_test_original))
        print('Specificity : ', spec_sum / len(y_test_original))

def binaryToDecimal(binary):
     
    binary1 = binary
    decimal, i, n = 0, 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    print(decimal) 
    return decimal

def numericdata(data):
    numericdatavalues = list()
    temprownumeric = list()
    for i in range(0, (len(data))):
        row = data[i]
        temprow = list()
        if(len(row) == 0):
            del data[i]
            continue
        floatvalues = [(float(item)) for item in row]
        temprownumeric.append(floatvalues)
    return temprownumeric

###############################################################################################################################
###############################################################################################################################
################################################################################################################################## 
from sklearn.model_selection import train_test_split
import time
import csv
start = time.time()
filename = 'C:\\Ayman\\PhDThesis\\iris_rank.txt'
# Set up input and output variables for the script
gpsTrack = open(filename, "r")
# Set up CSV reader and process the header
csvReader = csv.reader(gpsTrack)
# header = next(csvReader)

# Loop through the lines in the file and get each coordinate
for row in csvReader:
    data.append(row[0:4])
    labels.append(row[4:7])

numericlabels = numericlabels(labels)
numericdata_list = numericdata(data)
numericAlldata_list =numericdataandllabels(alldata)

enc=preprocessing.OneHotEncoder()


y=np.array(numericlabels)
X=np.array(numericdata_list)
numericAlldata_array=np.array(numericAlldata_list)

X,X_test,y,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



#######################################################################

net=initialize_network()

errors,net=training(net,500, 0.07,3)

pred=predict(net,X_test)


calcConfusion(pred,y_test)      
drawROC(y_test, pred, 1)     

# # output=np.argmax(pred)
print("end")
# print_network(net)
