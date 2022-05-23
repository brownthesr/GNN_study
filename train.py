from __future__ import division
from __future__ import print_function

# These are all python libraries that we want to utilize in our program
import time  # To track how long it takes
import argparse
import numpy as np  # because numpy is hecka important in everything
import os
import torch # this is the most important library for machine learning, it automates the backpropagation process and makes 
# matrix and tensor operations much easier
import torch.nn.functional as F # this guy is used to help us perform activation functions like ReLU or Sigmoid
import torch.optim as optim # this guy is the work horse behind backpropagation
from scipy import linalg
import matplotlib.pyplot as plt # this is just a standard plotting library

from utils import load_dataset, accuracy,np_to_torch # this is another set of code that we wrote that compiles our data
# and makes it easier to work with. It grabs our adjacency matrix, and our graph features/signals
from models import GCN # this is our main model that contains all the weight parameters and performs the matrix multiplication

# the following line of code obtains:
    # the adjacency matrix, adj, an nxn matrix(n = number of nodes) whose ith,jth entry indicates the presence of an edge between 
    #   nodes i and j. Basically it is a numerical way of representing our graph structure
    # The feature matrix, feature, an nxf matrix(f = number of different features) the first row contains all the features for
    #   for node 1, the second row contains all the features for node two, ect.
    # The labels of the nodes, labels, an nx1 matrix. Each entry indicates the group/class a node is in. Or its label
    #   For cora we have seven different groups. We have 7 different labels in this cora dataset
    # The training mask, training, with values [1,2,...,140] a vector. Basically this guy helps us sort which nodes we want to train
    #    on.
    # The validation mask, evaluating, with values [200,201,...,499] also a vector. Basically this guy helps us sort which nodes to
    #   perform evaluation on.
    # The testing mask, testing, with values [500,501,...1500] similarly a vector. This guy sorts which nodes to test on
    # The Graph, Network, This is a code representation of the graph using networkx, it is not necessary but I used it to 
    #   perturb the graph structure for robustness testing(I didn't perturb anything here tho)
# all of these things are built in the utils.py module, for a more detailed explanation of how they are computed look at that code   
adj, features, labels, training, evaluating, testing, Network = load_dataset(dataset="cora")


parser = argparse.ArgumentParser()# This is a convenient way to store all of our "hyper-parameters" or the parameters that
# we control over the network
parser.add_argument('--seed',type =int,default =42,help="random seed")# this is unimportant, basically the method to randomly sample numbers
parser.add_argument('--epochs',type = int, default=200, help="number of epochs to train for")# how many epochs, or full runs through our data
                                                                                            # that we want to perform
parser.add_argument("--lr", type = float,default=.01,help="initial learning rate")# the learning rate, related to backpropagation
                                                                                # in essence a measure of how fast vs how precise 
                                                                                # we want to learn parameters.
                                                                                # lower = slower learning but more precise
                                                                                # higher = faster learning but less precise
parser.add_argument("--weight_decay", type = float, default = 5e-4, help= " inital weight decay rate")# this helps us be precise and fast
                                                                                # by lowering the learning rate as we go
parser.add_argument("--hidden", type = int, default = 16, help= "number of hidden layers")# this is the number of hidden layers in the 
                                                                                # network
parser.add_argument("--dropout", type = float, default= .6, help = "dropout rate")# this is dropout, it is a little bit more advanced
                                                                            # but a huge problem with machine learning is overfitting
                                                                            # this reduces overfitting by randomly not using a portion
                                                                            # of the nodes when training the network
args = parser.parse_args()# this just grabs our hyperparameters

np.random.seed(args.seed)# the random seeding, really not important
torch.manual_seed(args.seed)

# this is our training regiment, this is called everytime we want to run the model through one epoch of training
def train(epoch):
    model.train()# this is relatively unimportant, but still needed. It just tells our model that we are starting training
    # so it gets ready to use dropout
    optimizer.zero_grad()# I will be honest, I have no clue what this does, it just is needed

    output = model(adj,features)# this runs forward propagation through our GNN. So it performs all the matrix operations
    # like Adj x Features x Weights.  It also runs activation functions on those outputs like Sigmoid(x) or Relu(x)
    # it then does this through multiple layers(2 in our case) to produce our output
    
    # this calculates the loss for you. In our case we use Negative-Log-Likelihood, this specific loss heavily penalizes wrong predictions
    # but other losses would work like MSE(mean squared error) or entropy
    # this is all only run on the training data, that is why we apply the mask "training" to it
    train_loss = F.nll_loss(output[training], labels[training])
    train_acc = accuracy(output[training],labels[training])# this calculates the accuracy of the model

    val_loss = F.nll_loss(output[evaluating], labels[evaluating])# this calculates the loss and accuracy of the model, only over the 
                                                              # validation data
    val_acc = accuracy(output[evaluating], labels[evaluating])

    train_loss.backward()# This computes the partial derivative information(gradients) to be used in backpropagation
    optimizer.step()#runs backpropagation for you

    if(epoch%5 == 0):# this makes it so we only print every 5 epochs
        print("epoch: {} training_loss: {} training_acc: {} val_loss: {} val_acc: {}".format(epoch,train_loss,train_acc,val_loss, val_acc))
    return train_loss.item(), val_loss.item(),train_acc.item(), val_acc.item() # returns our desired information for plotting

# this is our testing regiment it tests our model on nodes that have never been trained on    
def test(epoch):

    model.eval()# this is also relatively unimportant, basically it tells our model that we are testing now.
    # so the model will not run dropout
    output = model(adj,features)# just as before this runs the entire node dataset through our model just as in the training

    testing_loss = F.nll_loss(output[testing],labels[testing])# this now computes the loss and accuracy but only over the testing nodes
    testing_acc = accuracy(output[testing],labels[testing])

    # prints the desired information
    print("epoch: {} test_loss: {} test_acc: {}".format(epoch, testing_loss, testing_acc))
    return testing_acc # returns our information

# this creates our model as an Object, these different parameters are passed to the constructor of our GCN model
# the model is defined in a separate class under models.py. You can look in there for a more in depth explanation of 
# what it is exactly. But for our purposes right here, this just creates an item that is able to run a graph neural network.
# this specific GNN has input layers of size infeat(I believe in our case it is 1433) a hidden layer of size 16, and an output
# layer of size 7(which is the number of labels)
model = GCN(infeat=features.shape[1],hidfeat=args.hidden,outfeat=7,dropout=args.dropout)

# This is the optimizer, It is in charge of running backpropagation over our model. So because of that we pass the 
# learning rate into it, the models parameters(also called the weights), and our weight decay. Feel free to treat this 
# as a black box
optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

# this starts a timer so we know how long it takes to run
t_total = time.time()
# these are lists in which we will store different values of each over all the epochs.
# we store: train_loss- the average training loss over all training samples
        #   val_loss- the average validation loss over all validation samples
        #   train_acc - how accurate our model was on the training data, % of nodes correctly classified from training sample
        #   val_acc - how accurate our model was on the validation data
# The reason that we separate our data into training, validation, and testing is because we want to reduce overfitting 
# and seek a model that can accurately generalize to things it has never seen before.
# so we basically only run backpropagation across the loss of the training samples. Then we use how well it performs on the testing
# and validation data to measure how well it will generalize to unseen samples
train_losses, val_losses,train_accs, val_accs = [],[],[],[]

#we iterate over all epochs
for epoch in range(args.epochs):
    # this line of code does two things, it obtains values for our losses and accuracies for this specific epoch and
    # it runs backpropagation on our training data in the train() method
    new_train_loss, new_val_loss, new_train_acc, new_val_acc = train(epoch)

    # we just add our values to our list
    train_losses.append(new_train_loss)
    val_losses.append(new_val_loss)
    train_accs.append(new_train_acc)
    val_accs.append(new_val_acc)

# this runs a test on our model to see how it does to completely unseen data
test("final")

# prints the total time elapsed
print("total time elapsed: ", time.time()-t_total)

# this creates plots to graph how well/quickly our GNN learned over all the epochs
# this is not hard to do, text me if you have any questions
fig, axes = plt.subplots(1,2)
axes[0].plot(np.arange(args.epochs), train_losses)
axes[0].plot(np.arange(args.epochs), val_losses)
axes[0].set_title("loss_plot")
axes[0].set(xlabel="epochs", ylabel="loss")
axes[1].plot(np.arange(args.epochs), train_accs)
axes[1].plot(np.arange(args.epochs), val_accs)
axes[1].set_title("acc_plot")
axes[1].set(xlabel="epoch", ylabel="accuracy")
plt.show()



    
    
