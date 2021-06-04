import json
import os
from copy import deepcopy

import numpy as np

import torch
from torch_geometric.datasets import Planetoid, CoraFull

import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils.mapped_queue import MappedQueue


def greedy_partition(G, num_part=-1, weight=None, q_break=0):
    """
    Find k communities in graph using Clauset-Newman-Moore greedy modularity
    maximization.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists. Then join the pairs that least decrease modularity.

    Modified from
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities
    """

    # Count nodes and edges
    N = len(G.nodes())
    m = sum([d.get("weight", 1) for u, v, d in G.edges(data=True)])
    q0 = 1.0 / (2.0 * m)

    # Map node labels to contiguous integers
    label_for_node = {i: v for i, v in enumerate(G.nodes())}
    node_for_label = {label_for_node[i]: i for i in range(N)}

    # Calculate degrees
    k_for_label = G.degree(G.nodes(), weight=weight)
    k = [k_for_label[label_for_node[i]] for i in range(N)]

    # Initialize community and merge lists
    communities = {i: frozenset([i]) for i in range(N)}
    merges = []

    # Initial modularity
    partition = [[label_for_node[x] for x in c] for c in communities.values()]
    q_cnm = modularity(G, partition)

    # Initialize data structures
    # CNM Eq 8-9 (Eq 8 was missing a factor of 2 (from A_ij + A_ji)
    # a[i]: fraction of edges within community i
    # dq_dict[i][j]: dQ for merging community i, j
    # dq_heap[i][n] : (-dq, i, j) for communitiy i nth largest dQ
    # H[n]: (-dq, i, j) for community with nth largest max_j(dQ_ij)
    a = [k[i] * q0 for i in range(N)]
    dq_dict = {
        i: {
            j: 2 * q0 - 2 * k[i] * k[j] * q0 * q0
            for j in [node_for_label[u] for u in G.neighbors(label_for_node[i])]
            if j != i
        }
        for i in range(N)
    }
    dq_heap = [
        MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
    ]
    H = MappedQueue([dq_heap[i].h[0] for i in range(N) if len(dq_heap[i]) > 0])

    # Merge communities until we can't improve modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            dq, i, j = H.pop()
        except IndexError:
            break
        dq = -dq

        # Remove best merge from row i heap
        dq_heap[i].pop()

        # Push new row max onto H
        if len(dq_heap[i]) > 0:
            H.push(dq_heap[i].h[0])

        # If this element was also at the root of row j, we need to remove the
        # duplicate entry from H
        if dq_heap[j].h[0] == (-dq, j, i):
            H.remove((-dq, j, i))
            # Remove best merge from row j heap
            dq_heap[j].remove((-dq, j, i))
            # Push new row max onto H
            if len(dq_heap[j]) > 0:
                H.push(dq_heap[j].h[0])
        else:
            # Duplicate wasn't in H, just remove from row j heap
            dq_heap[j].remove((-dq, j, i))

        # Stop when change is non-positive 0
        if num_part <= 0 and dq <= q_break:
            break
        elif len(communities) == num_part:
            break

        # Perform merge
        communities[j] = frozenset(communities[i] | communities[j])
        del communities[i]
        merges.append((i, j, dq))
        # New modularity
        q_cnm += dq
        # Get list of communities connected to merged communities
        i_set = set(dq_dict[i].keys())
        j_set = set(dq_dict[j].keys())
        all_set = (i_set | j_set) - {i, j}
        both_set = i_set & j_set
        # Merge i into j and update dQ
        for k in all_set:
            # Calculate new dq value
            if k in both_set:
                dq_jk = dq_dict[j][k] + dq_dict[i][k]
            elif k in j_set:
                dq_jk = dq_dict[j][k] - 2.0 * a[i] * a[k]
            else:
                # k in i_set
                dq_jk = dq_dict[i][k] - 2.0 * a[j] * a[k]
            # Update rows j and k
            for row, col in [(j, k), (k, j)]:
                # Save old value for finding heap index
                if k in j_set:
                    d_old = (-dq_dict[row][col], row, col)
                else:
                    d_old = None
                # Update dict for j,k only (i is removed below)
                dq_dict[row][col] = dq_jk
                # Save old max of per-row heap
                if len(dq_heap[row]) > 0:
                    d_oldmax = dq_heap[row].h[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (-dq_jk, row, col)
                if d_old is None:
                    # We're creating a new nonzero element, add to heap
                    dq_heap[row].push(d)
                else:
                    # Update existing element in per-row heap
                    dq_heap[row].update(d_old, d)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d)
                else:
                    # We've updated an entry in this row, has the max changed?
                    if dq_heap[row].h[0] != d_oldmax:
                        H.update(d_oldmax, dq_heap[row].h[0])

        # Remove row/col i from matrix
        i_neighbors = dq_dict[i].keys()
        for k in i_neighbors:
            # Remove from dict
            dq_old = dq_dict[k][i]
            del dq_dict[k][i]
            # Remove from heaps if we haven't already
            if k != j:
                # Remove both row and column
                for row, col in [(k, i), (i, k)]:
                    # Check if replaced dq is row max
                    d_old = (-dq_old, row, col)
                    if dq_heap[row].h[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap[row].remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap[row]) > 0:
                            H.push(dq_heap[row].h[0])
                    else:
                        # Only update per-row heap
                        dq_heap[row].remove(d_old)

        del dq_dict[i]
        # Mark row i as deleted, but keep placeholder
        dq_heap[i] = MappedQueue()
        # Merge i into j and update a
        a[j] += a[i]
        a[i] = 0

    communities = [
       [label_for_node[i] for i in c] for c in communities.values()
    ]
    return sorted(communities, key=len, reverse=True)


def merge(x, communities, sizes):

    assert isinstance(sizes, list)
    assert len(communities) >= max(sizes)
    partitions = {}
    n = x.shape[1]

    while len(communities) >= min(sizes):

        print(len(communities))

        if len(communities) in sizes:
            partitions[len(communities)] = torch.zeros(x.shape[0], dtype=torch.int)
            for i, com in enumerate(communities):
                partitions[len(communities)][com] = i

        x_com = []
        for com in communities:
            x_com.append(x[com].mean(axis=0))
        x_com = torch.stack(x_com, dim=0)

        dist = torch.norm(
            x_com.reshape(1, -1, n) - x_com.reshape(-1, 1, n), dim=2
        )
        for i in range(len(communities)):
            dist[:, i] *= len(communities[i])
            dist[i, :] *= len(communities[i])

        dist = dist.numpy() + np.diag(np.ones(dist.shape[0]) * np.infty)

        closest_partitions = np.min(dist, axis=1)
        closest_idx = np.argmin(dist, axis=1)
        idx = np.argmin(closest_partitions)

        dist[idx][closest_idx[idx]] = np.infty
        dist[closest_idx[idx]][idx] = np.infty
        communities[idx].extend(communities[closest_idx[idx]])
        del communities[closest_idx[idx]]
        communities = sorted(communities, key=lambda c: len(c), reverse=True)

    return partitions


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
        data.g = nx.Graph()
        edges = [(int(i), int(j)) for i, j in data.edge_index.T]
        data.g.add_edges_from(edges)

        merge_size = data.num_classes
        data.max_part = data.num_classes
        graph = data.g.to_undirected()

        communities = greedy_partition(graph)
        data.partitions = merge(data.x, deepcopy(communities), list(range(1, merge_size + 1)))

    except UserWarning:
        pass

    return data


if __name__ == "__main__":
    for name in ["cora", "pubmed", "citeseer", "corafull"]:
        data = load_data(name)
        f = open("data/partitions_{}.json".format(name), "w", encoding='utf-8')
        part_save = {}
        for key, val in data.partitions.items():
            part_save[int(key)] = np.array(val).tolist()
        f.write(json.dumps(part_save, separators=(',', ':'), sort_keys=True))
        f.close()
