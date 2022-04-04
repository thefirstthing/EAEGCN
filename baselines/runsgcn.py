import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy, preprocess_data
from model import GCN, SGC, MLP, GraphSAGE
from GATdgl import GAT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cornell')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()
#device = 'cuda'

def gae_for(g, features, nclass):

    net = GCN(features.size()[1], args.hidden, nclass)
    ##net = GAT(g, in_dim=features.size()[1], hidden_dim=8, out_dim=6, num_heads=2, dropout=args.dropout)
    ##net = MLP(features.size()[1], args.hidden, nclass, 0)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # main loop
    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    for epoch in range(args.epochs):
        if epoch >= 1:
            t0 = time.time()

        net.train()
        logp = net(g, features)

        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_acc = accuracy(logp[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net(g, features)
        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= args.patience and args.dataset in ['cora', 'citeseer', 'pubmed']:
            print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)
        if epoch %100 == 0:
            print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
                epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))


    if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
        los.sort(key=lambda x: x[1])
        acc = los[0][-1]
        print('testacc')
        print(acc)
    else:
        los.sort(key=lambda x: -x[2])
        acc = los[0][-1]
        print('testacc')
        print(acc)


    return acc
#from utils import preprocess_nofeatures     TOC
if __name__ == '__main__':
    train_ratio=0.1
    acclist = []
    for i in range(6):

        g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, train_ratio)
        print(train_ratio)
        train_ratio += 0.1
        deg = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm

        acc=gae_for(g, features, nclass)
        acclist.append(acc)
    print(acclist)


