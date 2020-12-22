from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
ctx = mx.cpu()
# ctx = mx.gpu()
mx.random.seed(1)


:
batch_size = 64
num_inputs = 784
num_outputs = 10

def relu(X):
    return nd.maximum(X,nd.zeros_like(X))

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

def net(X, debug=False):
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3),
                                  num_filter=num_filter_conv_layer1)
    h1_activation = relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5),
                                  num_filter=num_filter_conv_layer2)
    h2_activation = relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Flattening h2 so that we can feed it into a fully-connected layer
    ########################
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Define the computation of the third (fully-connected) layer
    ########################
    h3_linear = nd.dot(h2, W3) + b3
    h3 = relu(h3_linear)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear
    
                                         
#######################
#  Set the scale for weight initialization and choose
#  the number of hidden units in the fully-connected layer
#######################
weight_scale = .01
num_fc = 128
num_filter_conv_layer1 = 20
num_filter_conv_layer2 = 50

W1 = nd.random_normal(shape=(num_filter_conv_layer1, 1, 3,3), scale=weight_scale, ctx=ctx)
b1 = nd.random_normal(shape=num_filter_conv_layer1, scale=weight_scale, ctx=ctx)

W2 = nd.random_normal(shape=(num_filter_conv_layer2, num_filter_conv_layer1, 5, 5),
                                                    scale=weight_scale, ctx=ctx)
b2 = nd.random_normal(shape=num_filter_conv_layer2, scale=weight_scale, ctx=ctx)

W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
b4 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]

for data, _ in train_data:
    data = data.as_in_context(ctx)
    break
conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3,3), num_filter=num_filter_conv_layer1)
print(conv.shape)


