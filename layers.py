import math

import torch

from torch.nn.parameter import Parameter # this is a class that we use to create our modifiable weigths an biases
from torch.nn.modules.module import Module# this class is needed to so that we can extend or inherit from it, to create our convolutional layer
import numpy as np # once again numpy is awesome

class GC(Module):# GC here stands for GraphConvolution. Basically it is just one layer in the model or one set of matrix multiplication
    # like Adj x Features x Weights
    def __init__(self, infeat, outfeat):# the constructor for our specific layer, it sets up our weights and biases
        super(GC,self).__init__()# this calls the constructor of the class that we inherit from, makes it so we can run backpropagation

        self.infeat = infeat# in features to the layer
        self.outfeat = outfeat# out features of the layer

        self.weights = Parameter(torch.FloatTensor(infeat,outfeat))# our weight matrix
        self.bias = Parameter(torch.FloatTensor(outfeat))# our bias for the layer

        self.reset_parameters()# this just randomizes the weight matrix, it makes converge faster
    
    def reset_parameters(self):# randomization of the weight matrix, makes it converge faster
        deviation = 1/(math.sqrt(self.weights.size(1)))
        self.weights.data.uniform_(-deviation,deviation)
        self.bias.data.uniform_(-deviation,deviation)

    def forward(self,input,adj):
        x = torch.mm(input,self.weights)
        # this is the python implemention of Features x Weights
        # so this is the neural network part of it
        x = torch.spmm(adj,x)
        # python implementation of adj x (Features x Weights)
        # this is spmm because it makes it faster
        # this is the aggregation operation

        x += self.bias
        # this adds the bias after

        return x# returns our final values

    def __repr__(self): # this is just for documentations sake
        return "this layer takes input of size :{} and outputs something of size: {}".format(self.infeat,self.outfeat)