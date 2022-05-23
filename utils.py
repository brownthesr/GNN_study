import numpy as np
import scipy.sparse as sp# this is used to put adjacency matrices in a more managaeable format
import torch
import networkx as nx

def encode_onehot(labels):
    # what one hot encoding is a mappping from the integers to a vector
    # it takes 0-> {1,0,0....}, it takes 1->{0,1,0,0,...}, 2->{0,0,1,0,0,...} where the length of each vector is the number of items 
    # to be classified. in essence it transforms a number i to a vector where all entries are zero but the ith entry is 1
    # this makes machine learning more effective.
    label_set = set(labels)

    label_dict = {j:np.identity(len(label_set))[i,:] for i, j in enumerate(label_set)}

    labels_onehot = np.array(list(map(label_dict.get,labels)),dtype=np.int32)
    return labels_onehot

def load_dataset(path="data/cora/", dataset="cora", directed = False):
    print("the {} dataset is being loaded".format(dataset))
    # this just informs us when the program starts

    info_id_fe_la = np.genfromtxt("{}{}.content".format(path,dataset), dtype=np.dtype(str))
    # this imports the feature and label information from cora.content

    features = sp.csr_matrix(info_id_fe_la[:,1:-1],dtype=np.float32)# obtains the feature information matrix
    labels = encode_onehot(info_id_fe_la[:,-1])# obtains the label information matrix and runs it through a onehot encoding
    # for an explanation of one hot encoding see the function of method above
    identifiers = np.array(info_id_fe_la[:,0],dtype = np.int32)
    # obtains the unique identifier for each node
    identifier_map = {j:i for i,j in enumerate(identifiers)}# labels each node with a number between 0 and 2708(number of nodes)

    edges_unordered = np.genfromtxt("{}{}.cites".format(path,dataset))# obtains the edge list from cora.cites
    edges = np.array(list(map(identifier_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    # this converts the edge list from being in terms of the node identifiers to being in terms of our node identifiers(0-2708)
    
    Network = nx.Graph()# this initiates a graph instance
    Network.add_nodes_from(identifier_map.values())# nodes to our graph
    Network.add_edges_from(edges)# add edges to our graph

    adj = nx.adjacency_matrix(Network)# this obtains an adjacency matrix from our graph
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)# this makes our adjacency matrix symmetric so its undirected
    adj = adj + sp.eye(adj.shape[0])# this adds self-loops to our graph
    adj2 = sparse_sp_to_torch(adj)# this just converts our sparse matrix to a torch matrix(the form we need it in for backpropagation)

    features = normalize(features)# normalizes our feature matrix

    training_set = 0
    evaluating_set = 0
    testing_set = 0
    if(dataset == "cora"):
        training_set = range(140)# this creates our training mask, says we just include the nodes from 0-139
        evaluating_set = range(200,500)# likewise this creates our validation mask to include nodes from 200-499
        testing_set = range(500,1500)# our testing mask included nodes from 500-1500
    
    # this converts all of our resulting values to torch tensors, this is needed for backpropagation
    features = torch.FloatTensor(np.array(features.todense()))# converts our feature to a torch matrix
    labels = torch.LongTensor(np.where(labels)[1])
    training_set = torch.LongTensor(training_set)
    evaluating_set = torch.LongTensor(evaluating_set)
    testing_set = torch.LongTensor(testing_set)

    return adj2, features, labels, training_set, evaluating_set, testing_set, Network

def normalize(x):
    # normalizes the matrix x
    rowsum = np.array(x.sum(1))
    row_inv = np.power(rowsum,-1).flatten()# we calculate this for the diagonal matrix
    row_inv[np.isinf(row_inv)] = 0
    rowdiags = sp.diags(row_inv)
    x = rowdiags.dot(x)
    return x

def accuracy(output, labels):# computes the accuracy of between the outputs and the actual labels
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def np_to_torch(adj,numnodes,numedges):
    # converts a numpy adjacency matrix to a torch adjacency matrix

    #we also add self edges 
    starting = np.array([u for u,v,a in adj])
    starting = np.append(starting, [v for u,v,a in adj])
    #we also add self edges 
    # We need this because the adj edges doesn't automatically create edges in both directions
    ending = np.array([v for u,v,a in adj])
    ending = np.append(ending,[u for u,v,a in adj])
    # this creates the corrosponding edge list
    arr = [starting,ending]
    indexes = torch.from_numpy(np.array(arr).astype(np.int64))

    values = torch.from_numpy(np.ones(numedges*2,dtype=np.float32))

    shape = torch.Size((numnodes,numnodes))
    return torch.sparse.FloatTensor(indexes,values,shape)

def sparse_sp_to_torch(sparse_mtx):
    # converts a sparse adjacency matrix to a torch adjacency matrix
    sparse_mtx = sparse_mtx.tocoo().astype(np.float32)

    indexes = torch.from_numpy(np.vstack((sparse_mtx.row,sparse_mtx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mtx.data)
    shape = torch.Size(sparse_mtx.shape)
    return torch.sparse.FloatTensor(indexes,values,shape)
