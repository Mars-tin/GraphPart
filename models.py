from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0,
                 activation="relu"):
        super(GCN, self).__init__()
        self.num_layers = 2
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, x, edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.activation(self.conv1(x, edge_index))


class GAT(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0,
                 activation="relu",
                 num_heads=4):
        super(GAT, self).__init__()
        self.num_layers = 2
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_size * num_heads, num_classes, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.activation(self.conv1(x, edge_index))
