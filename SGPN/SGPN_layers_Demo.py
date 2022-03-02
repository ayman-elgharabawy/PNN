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

from SGPN_Layers import SGPN_Layers
import os.path


###########################   Demo SGPNN3  ##############################
my_path = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(my_path, "..//Data//SGPNData//wine_iris_stock_3_3_5_small.csv")


#############################################################################################
filename = os.path.join(my_path, "..//Data//SGPNData//germn2005_2009_modified.csv")
sgpn2 = SGPN_Layers()
featuresno =31#34
hn = 100
scale = 20
fold =4
epochs = 50
noutputvalues = [2, 2]
noutputlist=[2,2]
lratelist=[0.07,0.07]
errorpredicted = sgpn2.loadData(filename=filename,  featuresno=featuresno,
                          noutputlist=noutputlist, noutputvalues=noutputvalues, epochs=epochs,
                          lratelist=lratelist, hn=hn, scale=scale, fold=fold)


# noutputlist=[3,3,5]
# trainederror,errorpredicted=loadData('wine_iris_stock_3_3_5',22,noutputlist,5000,0.05,200,1)
# plotTrainValidate(trainederror,errorpredicted)

# noutputlist=[130,130]
# loadData('RC_Final',18,noutputlist,2000,0.07,100)

# noutputlist=[5,5]
# loadData('germn2005_2009_modified',31,noutputlist,10000,0.07,50,20)

# noutputlist=[3,3,5]
# loadData('wine_iris_stock_3_3_5',22,noutputlist,2000,0.07,100)

# noutputlist=[16,3]
# loadData('wisconsin_iris_16_3',20,noutputlist,2000,0.07,100)


# noutputlist=[3,5]
# loadData('wine_stock_3_5',18,noutputlist,2000,0.07,100)


# noutputlist=[3,5]
# loadData('iris_stock_3_5',9,noutputlist,2000,0.07,100)

# noutputlist=[3,3]
# loadData('wine_iris_3_3',17,noutputlist,20000,0.07,25)

