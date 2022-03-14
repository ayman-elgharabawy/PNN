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
import PNN 
from PNN import PNN
import os.path

############################## Demo PNN ##############################

my_path = os.path.abspath(os.path.dirname(__file__))
pnn = PNN()
############################## IRIS Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//iris.txt")
# train_error = pnn.loadData(filename=path,featuresno= 4,labelno=3,ssteps=3,epochs=200,lrate=0.005,hn=100,Fold=3,useFold=True,scale=3,dropout=False,dropno=100) 
############################## WINE Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//wine.txt")
# train_error = pnn.loadData(filename=path,featuresno= 13,labelno=3,ssteps=3,epochs=100,lrate=0.005,hn=300,Fold=2,useFold=False,scale=2,dropout=False,dropno=100) 
############################## Vehicle Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//vehicle.txt")
# train_error = pnn.loadData(filename=path,featuresno= 18,labelno=4,ssteps=4,epochs=500,lrate=0.0008,hn=50,Fold=5,useFold=False,scale=3,dropout=False,dropno=100) 
############################## Stock Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//stock.txt")
# train_error = pnn.loadData(filename=path,featuresno= 5,labelno=5,ssteps=5,epochs=200,lrate=0.003,hn=300,Fold=5,useFold=True,scale=3,dropout=False,dropno=100) 
############################## Segment Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//segment.txt")
# train_error = pnn.loadData(filename=path,featuresno= 18,labelno=7,ssteps=7,epochs=100,lrate=0.007,hn=200,Fold=5,useFold=False,scale=3,dropout=False,dropno=100) 
# path = os.path.join(my_path, "..//Data//LRData//elevators.txt")
# train_error = pnn.loadData(filename=path,featuresno= 9,labelno=9,ssteps=9,epochs=100,lrate=0.003,hn=100,Fold=5,useFold=False,scale=3,dropout=False,dropno=100)
# path = os.path.join(my_path, "..//Data//LRData//glass.txt")
# train_error = pnn.loadData(filename=path,featuresno= 9,labelno=6,ssteps=6,epochs=500,lrate=0.005,hn=80,Fold=5,useFold=False,scale=3,dropout=False,dropno=100)

# ############################## Fried Dataset ##############################
# path = os.path.join(my_path, "LRData//fried.txt")
# train_error = pnn.loadData(filename=path,featuresno= 9,labelno=5,ssteps=5,epochs=100,lrate=0.005,hn=100,Fold=5,useFold=False,scale=3)

# ############################## pendigits Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//pendigits.txt")
# train_error = pnn.loadData(filename=path,featuresno= 16,labelno=10,ssteps=10,epochs=100,lrate=0.007,hn=350,Fold=5,useFold=False,scale=3,dropout=False,dropno=100) 
# ############################## Vowl Dataset ##############################
path = os.path.join(my_path, "..//Data//LRData//vowel.txt")
train_error = pnn.loadData(filename=path,featuresno= 10,labelno=11,ssteps=11,epochs=1000,lrate=0.003,hn=200,Fold=5,useFold=False,scale=3,dropout=False,dropno=100) 

# ############################## Housing Dataset ##############################
# path = os.path.join(my_path, "..//Data//LRData//housing.txt")
# train_error = pnn.loadData(filename=path,featuresno= 6,labelno=6,ssteps=6,epochs=100,lrate=0.005,hn=100,Fold=5,useFold=True,scale=3,dropout=False,dropno=50) 


# path = os.path.join(my_path, "..//Data//LRData//fried.txt")
# train_error = pnn.loadData(filename=path,featuresno= 9,labelno=5,ssteps=5,epochs=100,lrate=0.005,hn=100,Fold=5,useFold=False,scale=3,dropout=False,dropno=50)
