import torch

import torch.nn as nn
import torch.nn.functional as F


# Define a GAT layer
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)  # eq. 1
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)  # eq. 2
        self.g.update_all(self.message_func, self.reduce_func)  # eq. 3 and 4
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        self.dropout = dropout



    def forward(self, g, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(h, 1)

'''
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh


def load_core_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, mask




import time
import numpy as np

g, features, labels, mask = load_core_data()

net = GAT(g, in_dim=features.size()[1], hidden_dim=8, out_dim=7, num_heads=8, dropout=args.dropout)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(300):
    if epoch >= 3:
        t0 = time.time()
        t0 = time.time()

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))
'''