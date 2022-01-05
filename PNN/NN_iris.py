import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os.path
#---------------------------------LOADING DATASET AND DATA PREPROCESSING-------------------------------------------#
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "..//Data//IRIS.csv")

data=pd.read_csv(path)

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
StandardScaler=StandardScaler()
X=StandardScaler.fit_transform(X)

label_encoder=LabelEncoder()
Y=label_encoder.fit_transform(Y)
Y=Y.reshape(-1,1)
enc=preprocessing.OneHotEncoder()
enc.fit(Y)
onehotlabels=enc.transform(Y).toarray()
Y=onehotlabels
#-----------------------------------------SPLITTING THE DATA INTO TRAINING AND TESTING---------------------------------------------#

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#----------------------------------------DEFINING VARIOUS ACTIVATION FUNCTIONS AND THEIR DERIVATIVES-------------------------------#
def ReLU(x):
	return (abs(x.astype(float))+x.astype(float))/2

def ReLU_derivative(x):
	y=x
	np.piecewise(y,[ReLU(y)==0,ReLU(y)==y],[0,1])
	return y

def tanh(x):
	return np.tanh(x.astype(float))

def tanh_derivative(x):
	return 1-np.square(tanh(x))

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)

#-------------------------------------CONSTRUCTING OUR NEURAL NETWORK CLASS/STRUCTURE-----------------------------------------#
class Neural_Network:
	def __init__(self,x,y,h):
		self.input=x
		self.weights1=np.random.randn(self.input.shape[1],h)
		self.weights2=np.random.randn(h,3)
		self.y=y
		self.output=np.zeros(y.shape)

	def FeedForward(self):
		self.layer1=ReLU(np.dot(self.input,self.weights1))
		self.output=sigmoid(np.dot(self.layer1,self.weights2))

	def BackPropogation(self):
		m=len(self.input)
		d_weights2=-(1/m)*np.dot(self.layer1.T,(self.y-self.output)*sigmoid_derivative(self.output))
		d_weights1 =-(1/m)*np.dot(self.input.T, (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * ReLU_derivative(self.layer1)))
		self.weights2=self.weights2 - lr*d_weights2
		self.weights1=self.weights1 - lr*d_weights1
	def predict(self,X):
		self.layert_1=ReLU(np.dot(X,self.weights1))
		return sigmoid(np.dot(self.layert_1,self.weights2))

#------------------------------------------{TRAINING OUR NETWORK OVER THE TRAINING DATA AND---------------------------------# 
#------------------------------------------EVALUATING VARIOUS PARAMETERS AFTER EACH EPOCH}----------------------------------#

epochs=10000
lr=0.5
n=len(X_test)
m=len(X)
nn1=Neural_Network(X_train,Y_train,1)
for i in range(epochs):
	nn1.FeedForward()
	y_predict_train=enc.inverse_transform(nn1.output.round())
	y_predict_test=enc.inverse_transform(nn1.predict(X_test).round())
	y_train=enc.inverse_transform(Y_train)
	y_test=enc.inverse_transform(Y_test)
	train_accuracy=(m-np.count_nonzero(y_train-y_predict_train))/m
	test_accuracy=(n-np.count_nonzero(y_test-y_predict_test))/n
	nn1.BackPropogation()
	cost=(1/m)*np.sum(np.square(nn1.y-nn1.output))
	print("Epoch {}/{} ==============================================================:- ".format(i+1,epochs))
	print("MSE_Cost: {} , Train_Accuracy: {} , Test_Accuracy: {} ".format(cost,train_accuracy,test_accuracy))

output=nn1.predict(X_test)
Y_predict=enc.inverse_transform(output.round())
Y_test=enc.inverse_transform(Y_test)
accuracy=(len(Y_predict)-np.count_nonzero(Y_test-Y_predict))/len(Y_predict)
print("The accuracy of the model is {}".format(accuracy))