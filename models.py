import torch.nn as nn# this is a set of all the modules we need for the code to run correctly

import torch.nn.functional as F# same as in train.py

from layers import GC# this is a module that we created that holds the GraphConvolutional layers,
# they are basically just a representation of a single matrix multiplication of 
# adj x features x weights

import numpy as np# numpy is awesome

class GCN(nn.Module): # For those who don't know python. (nn.Module) means that our class GCN inherits or extends from the 
    # class nn.Module. This is important because it makes it so that we can run backpropagation/forwardpropagation on any objects that we 
    # create from this class

    def __init__ (self, infeat, hidfeat, outfeat, dropout):# this is the constructor, or initializer. Stuff online can probably explain it
                                                            # better than I can
        super(GCN,self).__init__()# this calls the constructor of the class we inherited from(nn.Module) basically it sets stuff up
                                # for backpropagation

        self.layer1 = GC(infeat, hidfeat)# this creates a single Graph Convolutional Layer, or one matrix multiplication that we talked
        # about above
        self.layer2 = GC(hidfeat, outfeat)# this is our second layer
        self.dropout = dropout# this is our dropout


    def forward(self, adj, x):# this is run everytime that we want to run data through our GNN or forward propagation
        x = F.relu(self.layer1(x,adj))# this runs data through our first layer then performs a ReLU activation elementwise
        # on the resulting matrix

        x = F.dropout(x, self.dropout, training = self.training)# this performs dropout, which means to randomly zero out some
        # of the entries in our new matrix, this helps reduce overfitting. For a more in depth explanation the internet is probably
        # a better source

        x = self.layer2(x,adj)# this runs it through a second layer of GNN

        return F.log_softmax(x,dim=1)# our second activation is called a log softmax. This function has some nice properties
        # that make this specific GNN learn better. For more information you can look up the Log-Softmax activation function
