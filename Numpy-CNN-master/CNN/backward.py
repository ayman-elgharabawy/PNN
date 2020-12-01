'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''

import numpy as np

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################
        
def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias



def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout

def dSSS(xi,nlabel,bx=1): 
    derivative2 = 0
    s=100
    b=100/bx
    t=200
    for i in range(nlabel):
        xx=s-((i*t)/(nlabel-1))
        derivative2 +=0.5*(1-np.power(np.tanh((-b*(xi))-(xx)),2))
    derivative2=-1*derivative2     
    derivative2= derivative2+(nlabel*0.5)  
    return derivative2


def DSpearman(output,expected):
    n=len(expected)
    dif=0
    diflist=np.array([])
    deflist=np.array([])
    for i in range (n):
        diflist=np.append(diflist,[2*(output[i])-2*(expected[i])])
        dif+=2*(output[i])-2*(expected[i])
      
    den=n*(np.power(n,2)-1)
    for dd in diflist:
        deflist=np.append(deflist,[((dd)/(den))])
    return deflist  
