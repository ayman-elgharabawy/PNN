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
import NN 
from NN import NN
import os.path

############################## Demo PNN ##############################

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "..//Data//iris_classification.txt")

nn = NN()
train_error = nn.loadData(filename=path,featuresno= 4,labelno=3,epochs=5000,lrate=0.05,hn=100,Fold=10,useFold=False) 

