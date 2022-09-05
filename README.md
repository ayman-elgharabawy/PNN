# Preference Neural Network (PNN) 
![ScreenShot](/Images/PNN.png)
PNN is a native ranker neural network implemented for label ranking using Spearman ranking error function and Positive/Smooth Staircase activation function (SS) and (PSS) to enhance the prediction probability to produce almost discrete value without data freedom between layers, thus it use one middle layer for learning.

# Subgroup Preference Neural Network (SGPNN)

<p align="center">
<img src="/Images/MAFN.png" width="600" height="300">
</p>

![ScreenShot](/Images/SGPNN.png)

SGPNN is extended PNN to ranke subgroup of labels using one learning model. the subgroups are combined from multiple domains to find a hidden relations between these groups.

These two networks use a new type of multi-values activation functions, Positive smooth staircase (PSS) and Smooth Staircase (SS) employed for ranking
![ScreenShot](/Images/SS.png)


## Wolframe Mathematica Positive Smotth Staircase (PSS) and Smooth Staircase (SS) 

The Positive Smooth Staircase (PSS) function Mathematica code where # steps = 4 and step width =1  is:

**Manipulate[Plot[(Sum[-0.5*Tanh[-100*(x - (w*i))], {i, 0, n - 1}]) + (n/2), {x, -1, 6}], {n, 4}, {w, 1}]**

![ScreenShot](/Images/PSS_wm.png)

Smooth Staircase (SS) function Mathematica code where # steps = 5 and boundaries between -1 and 1  is:

**Manipulate[Plot[(-0.5*Sum[( Tanh[(-x * 100)/b + c*(1 - (2*i/(n - 1)))] ), {i, 0,  n - 1}]) + ((n)/2), {x, -4, 4}], {n, 5}, {c, -100}, {b, 1}]**

![ScreenShot](/Images/SS_wm.png)


Smooth Staircase (SS) function for regression Mathematica code where # steps n = 60 and decimal value s = 0.1  is:

**Manipulate[Plot[-0.5/s *(Sum[Tanh[(-20s*x)+(10*n)*((2*i/(n-1)) -1)],{i, 0, n - 1} ]-n), {x, -4, 4}],{s, 10,10} ,{n,60}]**

## Python example 

Using pythin +3.7

pnn = PNN()

train_error = pnn.loadData(filename=path,featuresno= 4,labelno=3,ssteps=2,epochs=500,lrate=0.005,hn=100,Fold=10,useFold=False) 


## References 
For feedback kindly connunicate using my email aaaeg@hotmail.com

Video Demo available at  https://drive.google.com/drive/folders/1yxuqYoQ3Kiuch-2sLeVe2ocMj12QVsRM?usp=sharing

Please Cite using the following links:

Elgharabawy, A.; Prasad, M.; Lin, C.-T. Subgroup Preference Neural Network. Sensors 2021, 21, 6104. https://doi.org/10.3390/s21186104

Elgharabawy, A.; Prasad, M.; Lin, C.-T. Preference Neural Network. 2021 arXiv:1904.02345 [cs.LG]
