from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import cluster

from dgl import function as fn

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class Cluster:
    """
    Kmeans Clustering
    """
    def __init__(self, n_clusters, n_dim, seed,
                 implementation='sklearn',
                 init='k-means++',
                 device=torch.cuda.is_available()):

        assert implementation in ['sklearn', 'faiss', 'cuml']
        assert init in ['k-means++', 'random']

        self.n_clusters = n_clusters
        self.n_dim = n_dim
        self.implementation = implementation
        self.initialization = init
        self.model = None

        if implementation == 'sklearn':
            self.model = cluster.KMeans(n_clusters=n_clusters, init=init, random_state=seed)
        elif implementation == 'faiss':
            import faiss
            self.model = faiss.Kmeans(n_dim, n_clusters, niter=20, nredo=10, seed=seed, gpu=device != 'cpu')
        elif implementation == 'cuml':
            import cuml
            if init == 'k-means++':
                init = 'scalable-kmeans++'
            self.model = cuml.KMeans(n_dim, n_clusters, random_state=seed, init=init, output_type='numpy')
        else:
            raise NotImplemented

    def train(self, x):
        if self.implementation == 'sklearn':
            self.model.fit(x)
        elif self.implementation == 'faiss':
            if self.initialization == 'kmeans++':
                init_centroids = self._kmeans_plusplus(x, self.n_clusters).cpu().numpy()
            else:
                init_centroids = None
            self.model.train(x, init_centroids=init_centroids)
        elif self.implementation == 'cuml':
            self.model.fit(x)
        else:
            raise NotImplemented

    def predict(self, x):
        if self.implementation == 'sklearn':
            return self.model.predict(x)
        elif self.implementation == 'faiss':
            _, labels = self.model.index.search(x, 1)
            return labels
        else:
            raise NotImplemented

    def get_centroids(self):
        if self.implementation == 'sklearn':
            return self.model.cluster_centers_
        elif self.implementation == 'faiss':
            return self.model.centroids
        elif self.implementation == 'cuml':
            return self.model.cluster_centers_
        else:
            raise NotImplemented

    def get_inertia(self):
        if self.implementation == 'sklearn':
            return self.model.inertia_
        else:
            raise NotImplemented

    @staticmethod
    def _kmeans_plusplus(X, n_clusters):
        """
        K-means++ initialization in PyTorch for Faiss.

        Modified from sklearn version of implementation.
        https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/cluster/_kmeans.py
        """

        n_samples, n_features = X.shape

        # Set the number of local seeding trials if none is given
        n_local_trials = 2 + int(np.log(n_clusters))

        # Pick first center randomly and track index of point
        center_id = torch.randint(n_samples, (1,)).item()
        centers = [X[center_id]]

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = torch.cdist(X, X[center_id].unsqueeze(dim=0)).pow(2).squeeze()
        current_pot = closest_dist_sq.sum()

        # Pick the remaining n_clusters-1 points
        for c in range(1, n_clusters):
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = torch.rand(n_local_trials).to(current_pot.device) * current_pot
            candidate_ids = torch.searchsorted(torch.cumsum(closest_dist_sq.flatten(), dim=0), rand_vals)

            # Numerical imprecision can result in a candidate_id out of range
            torch.clip(candidate_ids, min=None, max=closest_dist_sq.shape[0] - 1, out=candidate_ids)

            # Compute distances to center candidates
            distance_to_candidates = torch.cdist(X[candidate_ids].unsqueeze(dim=0), X).pow(2).squeeze()

            # update closest distances squared and potential for each candidate
            torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(dim=1)

            # Decide which candidate is the best
            best_candidate = torch.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            # Permanently add best center candidate found in local tries
            centers.append(X[best_candidate])

        centers = torch.stack(centers, dim=0).to(dtype=X.dtype)
        return centers


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, batchnorm=False, activation="relu"):
        super(GCN, self).__init__()

        assert activation in ["relu", "elu"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        if batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            if batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        x = self.embed(x, adj_t)
        x = self.convs[-1](x, adj_t)
        return x

    def embed(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if len(self.bns) > 0:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SAGE(torch.nn.Module):
    """
    GraphSAGE
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, batchnorm=False, activation="relu"):
        super(SAGE, self).__init__()

        assert activation in ["relu", "elu"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        if batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        x = self.embed(x, adj_t)
        x = self.convs[-1](x, adj_t)
        return x

    def embed(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if len(self.bns) > 0:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(torch.nn.Module):
    """
    Graph Attention Network
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads,
                 num_layers=2, dropout=0.5, batchnorm=False, activation="relu"):
        super(GAT, self).__init__()

        assert activation in ["relu", "elu"]

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, bias=False))
        self.bns = torch.nn.ModuleList()
        if batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.num_layers = num_layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, bias=False))
            if batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GATConv(hidden_channels * num_heads, out_channels, heads=num_heads, bias=False))

        self.dropout = dropout
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        x = self.embed(x, adj_t)
        x = self.convs[-1](x, adj_t)
        return x

    def embed(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if len(self.bns) > 0:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
