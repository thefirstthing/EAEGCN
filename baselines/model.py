import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn

from dgl.nn.pytorch import GraphConv, SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,  in_feats, hidden_size, num_classes, n_layers):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        self.aggregator_type ='gcn'
        ##self.activation = None

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, self.aggregator_type))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(hidden_size, hidden_size, self.aggregator_type))
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, self.aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers):
                #h = self.activation(h)
                h = self.dropout(h)
        return F.log_softmax(h, 1)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return  F.log_softmax(h, 1)


class SGC(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(SGC, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        #h = torch.relu(h)
        h = self.conv2(g, h)
        return  F.log_softmax(h, 1)


# The first layer transforms input features of size of 5 to a hidden size of 5.
# The second layer transforms the hidden layer and produces output features of
# size 2, corresponding to the two groups of the karate club.



#
class MLP(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout):
        super(MLP, self).__init__()
        self.ln1 =nn.Linear(in_feats, hidden_size)
        self.ln2= nn.Linear(hidden_size, num_classes)
        self.dropout = dropout
    def forward(self, g, inputs):
        h = F.dropout(inputs, p=self.dropout, training=self.training)
        h = self.ln1(h)
        h = torch.relu(h)
        h = self.ln2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(h, 1)