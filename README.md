# Preference Neural Network (PNN)

<p align="center">
<img src="/Images/PNN.png" width="400" height="200">
</p>

PNN is a native ranker neural network implemented for label ranking using Spearman ranking error function and Positive/Smooth Staircase activation function (SS) and (PSS) to enhance the prediction probability to produce almost discrete value without data freedom between layers, thus it use one middle layer for learning.

# Subgroup Preference Neural Network (SGPNN)

<p align="center">
<img src="/Images/MAFN.png" width="400" height="200">
</p>


<p align="center">
<img src="/Images/SGPNN.png" width="350" height="200">
</p>

SGPNN is extended PNN to rank subgroup of labels using one learning model. the subgroups are combined from multiple domains to find a hidden relations between these groups.

These two networks use a new type of multi-values activation functions, Positive smooth staircase (PSS) and Smooth Staircase (SS) employed for ranking

<p align="center">
<img src="/Images/eq_ss.png" width="350" height="200">
</p>


Smooth Staircase (SS) function where # steps = 5 and boundaries between -1 and 1  is:

**Manipulate[Plot[(-(s/2)*(Sum[( Tanh[(c*(b-x-(w*i)))] ), {i, 0, n - 1}]-(1)) ), {x, -4, 4}], {n, 5},{s,1,1000}, {c, 100}, {b, 2},{w,1}]**

<p align="center">
<img src="/Images/ss.png" width="850" height="400">
</p>

Smooth Staircase (SS) function for regression value up to 2 decimal value where # steps = 5 and boundaries between -1 and 1  is:

<p align="center">
<img src="/Images/ss_001.png" width="850" height="400">
</p>

## Python example

Using python +3.7

pnn = PNN()

train_error = pnn.loadData(filename=path,featuresno= 4,labelno=3,ssteps=2,epochs=500,lrate=0.005,hn=100,Fold=10,useFold=False)


## References
For feedback kindly communicate using my email aaaeg@hotmail.com

Video Demo available at  https://drive.google.com/drive/folders/1yxuqYoQ3Kiuch-2sLeVe2ocMj12QVsRM?usp=sharing

Please Cite using the following links:

A. Elgharabawy, M. Prasad and C. -T. Lin, "Preference Neural Network," in IEEE Transactions on Emerging Topics in Computational Intelligence, doi: 10.1109/TETCI.2023.3268707.

A. Elgharabawy, M. Prasad, and C.-T. Lin, “Subgroup Preference Neural Network,” Sensors, vol. 21, no. 18, p. 6104, Sep. 2021, doi: 10.3390/s21186104.


