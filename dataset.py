import codecs
import json
import os
import numpy as np

import dgl
import torch
import networkx as nx

from torch_geometric.datasets import Planetoid, CoraFull


def load_data(name="cora"):

    assert name in ["cora", "pubmed", "citeseer", "corafull"]

    path = os.path.join("data", name)
    if name == "corafull":
        dataset = CoraFull(root=path, transform=None)
    else:
        dataset = Planetoid(root=path, name=name, transform=None)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.adj = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.int)
    data.adj[data.edge_index[0], data.edge_index[1]] = 1

    try:
        if name == 'cora':
            # n = 2708
            data.params = {
                'age': [0.025, 0.025, 0.95],
            }
        elif name == 'pubmed':
            # n = 19717
            data.params = {
                'age': [0.05, 0.05, 0.9],
            }
        elif name == 'citeseer':
            # n = 3327
            data.params = {
                'age': [0.35, 0.35, 0.3],
            }
        else:
            # n = 19793
            data.params = {
                'age': [0.05, 0.05, 0.9],
            }

        data.g = nx.Graph()
        edges = [(int(i), int(j)) for i, j in data.edge_index.T]
        data.g.add_edges_from(edges)

        # Get x_prop for L = 2
        A = data.adj + torch.eye(data.adj.shape[0])
        D = torch.diag(A.sum(dim=0) ** (-1 / 2))
        A_norm = torch.sparse.mm(torch.sparse.mm(D, A), D)
        data.x_prop = A_norm
        data.x_prop = torch.sparse.mm(data.x_prop, A_norm)
        data.x_prop = torch.sparse.mm(data.x_prop, data.x)

        data.g = dgl.from_networkx(data.g)

        data.max_part = data.num_classes
        if name == 'pubmed':
            data.max_part = 6	# 6 = 3*2
        if name == 'citeseer':
            data.max_part = 7	# 7 = 70/10

        data.partitions = {}
        obj_text = codecs.open("data/partitions/partitions_{}.json".format(name), 'r', encoding='utf-8').read()
        part_dict = json.loads(obj_text)
        for key, val in part_dict.items():
            data.partitions[int(key)] = torch.tensor(val)

    except UserWarning:
        pass
    return data
