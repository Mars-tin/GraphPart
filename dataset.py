import json
import codecs
import os

import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CoraFull, Coauthor

from ogb.nodeproppred import PygNodePropPredDataset

import networkx as nx
from partition import GraphPartition


def load_data(name="cora", read=True, save=False,
              transform=T.ToSparseTensor(),
              pre_compute=True, verbose=False):

    assert name in ["cora", "pubmed", "citeseer", "corafull",
                    "cs", "physics", 'arxiv']

    path = os.path.join("data", name)
    if name == 'cora' or name == 'pubmed' or name == 'citeseer':
        dataset = Planetoid(root=path, name=name, transform=transform)
    elif name == "corafull":
        dataset = CoraFull(root=path, transform=transform)
    elif name == "cs" or name == "physics":
        dataset = Coauthor(root=path, name=name, transform=transform)
    elif name == 'arxiv':
        dataset = PygNodePropPredDataset(root=path, name='ogbn-' + name, transform=transform)
    else:
        raise NotImplemented

    data = dataset[0]
    if not hasattr(data, 'num_classes'):
        data.num_classes = dataset.num_classes
    data.adj_t = data.adj_t.to_symmetric() if not isinstance(data.adj_t, torch.Tensor) else data.adj_t
    data.max_part = data.num_classes

    try:
        if name == 'cora':
            data.max_part = 7
            data.params = {'age': [0.05, 0.05, 0.9]}
        elif name == 'pubmed':
            data.max_part = 8
            data.params = {'age': [0.15, 0.15, 0.7]}
        elif name == 'citeseer':
            data.max_part = 14
            data.params = {'age': [0.35, 0.35, 0.3]}
        elif name == 'corafull':
            data.max_part = 7
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'cs':
            data.max_part = 6
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'physics':
            data.max_part = 5
            data.params = {'age': [0.1, 0.1, 0.8]}
        elif name == 'arxiv':
            data.max_part = 9
            data.params = {'age': [0.1, 0.1, 0.8]}
        else:
            raise NotImplemented

        if not hasattr(data, 'g'):
            edges = [(int(i), int(j)) for i, j in zip(data.adj_t.storage._row,
                                                      data.adj_t.storage._col)]
            data.g = nx.Graph()
            data.g.add_edges_from(edges)

        if read:
            filename = "data/partitions.json"
            if os.path.exists(filename):
                data.partitions = {}
                obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
                part_dict = json.loads(obj_text)
                data.max_part = part_dict[name]['num_part']
                data.partitions[data.max_part] = torch.tensor(part_dict[name]['partition'])
            else:
                print('Partition file not found!')
                raise NotImplemented

        else:
            graph = data.g.to_undirected()
            graph_part = GraphPartition(graph, data.x, data.max_part, name, log=False)

            communities = graph_part.clauset_newman_moore(weight=None)
            sizes = ([len(com) for com in communities])
            threshold = 1/3
            if min(sizes) * len(sizes) / len(data.x) < threshold:
                data.partitions = graph_part.agglomerative_clustering(communities)
            else:
                sorted_communities = sorted(communities, key=lambda c: len(c), reverse=True)
                data.partitions = {}
                data.partitions[len(sizes)] = torch.zeros(data.x.shape[0], dtype=torch.int)
                for i, com in enumerate(sorted_communities):
                    data.partitions[len(sizes)][com] = i

        if verbose:
            from collections import Counter
            for num_part, partitions in data.partitions.items():
                purity = []
                for i in range(num_part):
                    y_part = data.y[np.where(partitions == i)[0]]
                    y_part = y_part.numpy().astype(int)
                    label = max(k for k, v in Counter(y_part).items()
                                if v == max(Counter(y_part).values()))
                    hit = y_part[np.where(y_part == label)[0]]
                    purity.append([label, len(hit) / len(y_part)])
                print(num_part, purity)

        if save and data.partitions is not None:
            f = open("data/partitions/{}.json".format(name), "w", encoding='utf-8')
            part_save = {}
            for key, val in data.partitions.items():
                part_save[int(key)] = np.array(val).tolist()
            f.write(json.dumps(part_save, separators=(',', ':'), sort_keys=True))
            f.close()

        edges = [(int(i), int(i)) for i in range(data.num_nodes)]
        data.g.add_edges_from(edges)

        if pre_compute:
            feat_dim = data.x.size(1)
            conv = GCNConv(feat_dim, feat_dim, cached=True, bias=False)
            conv.lin.weight = torch.nn.Parameter(torch.eye(feat_dim))
            with torch.no_grad():
                data.aggregated = conv(data.x, data.adj_t)
                data.aggregated = conv(data.aggregated, data.adj_t)

    except UserWarning:
        pass

    return data


if __name__ == "__main__":
    for name in ["cora", "pubmed", "citeseer", "corafull", "cs", "physics", 'arxiv']:
        data = load_data(name=name, read=False, save=True)
