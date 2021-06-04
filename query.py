import numpy as np
from copy import deepcopy

import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank

import torch
import torch.nn.functional as F
import torch.optim as optim


class ActiveLearningAgent:
    """
    Base class.
    """

    def __init__(self, data, model, seed, args):
        self.round = 0
        self.data = data
        self.model = model
        self.seed = seed
        self.args = args
        self.retrain = args.retrain
        self.clf = None
        self.x_prop = None
        self.num_centers = args.num_centers
        self.num_parts = -1

    def query(self, b):
        pass

    def update(self, train_mask):
        self.data.train_mask = train_mask
        self.round += 1

    def train(self):
        if self.retrain:
            self.clf = deepcopy(self.model).to(self.args.device)
        else:
            self.clf = self.model.to(self.args.device)
        optimizer = optim.Adam(
            self.clf.parameters(), lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        for epoch in range(self.args.epochs):
            self.clf.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(
                self.clf(self.data.x, self.data.edge_index)[self.data.train_mask],
                self.data.y[self.data.train_mask])
            if self.args.verbose == 2:
                print('Epoch {:03d}: Training loss: {:.4f}'.format(epoch, loss))
            loss.backward()
            optimizer.step()

    def predict(self):
        self.clf.eval()
        logits = self.clf(self.data.x, self.data.edge_index)
        y_pred = logits.max(1)[1].cpu()
        y_true = self.data.y.cpu()
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        if self.args.verbose == 2:
            print('Macro-f1 score: {:.4f}'.format(f1))
        return f1

    def transform(self, method='prop'):

        if method == 'prop':
            if self.clf.num_layers > 2:
                if self.x_prop is None:
                    A = self.data.adj + torch.eye(self.data.adj.shape[0], device=self.args.device)  # A + I
                    D = torch.diag(A.sum(dim=0) ** (-1 / 2))  # (D + I)^(-1/2)
                    A_norm = torch.sparse.mm(torch.sparse.mm(D, A), D)
                    self.x_prop = A_norm
                    for i in range(self.clf.num_layers - 1):
                        self.x_prop = torch.sparse.mm(self.x_prop, A_norm)
                    self.x_prop = torch.sparse.mm(self.x_prop, self.data.x)
                return self.x_prop
            return self.data.x_prop

        elif method == 'embed':
            with torch.no_grad():
                embed = self.clf.encode(self.data.x, self.data.edge_index)
            return embed

        else:
            return self.data.x

    def get_n_cluster(self, b, partitions, x_embed=None, method='default'):

        if method == 'size':
            part_size = []
            for i in range(self.num_parts):
                part_size.append(len(np.where(partitions == i)[0]))
            part_size = np.rint(b * np.array(part_size) / sum(part_size)).astype(int)
            part_size = np.maximum(self.num_centers, part_size)
            i = 0
            while part_size.sum() - b != 0:
                if part_size.sum() - b > 0:
                    i = self.num_parts - 1 if i <= 0 else i
                    while part_size[i] <= 1:
                        i -= 1
                    part_size[i] -= 1
                    i -= 1
                else:
                    i = 0 if i >= self.num_parts else i
                    part_size[i] += 1
                    i += 1

        elif method == 'inertia':
            part_size = []
            for i in range(self.num_parts):
                part_id = np.where(partitions == i)[0]
                x = x_embed[part_id]
                kmeans = KMeans(n_clusters=1, init='k-means++')
                kmeans.fit(x.cpu())
                inertia = kmeans.inertia_
                part_size.append(inertia)

            part_size = np.rint(b * np.array(part_size) / sum(part_size)).astype(int)
            part_size = np.maximum(self.num_centers, part_size)
            i = 0
            while part_size.sum() - b != 0:
                if part_size.sum() - b > 0:
                    i = self.num_parts - 1 if i <= 0 else i
                    while part_size[i] <= 1:
                        i -= 1
                    part_size[i] -= 1
                    i -= 1
                else:
                    i = 0 if i >= self.num_parts else i
                    part_size[i] += 1
                    i += 1

        else:
            part_size = [b // self.num_parts for _ in range(self.num_parts)]
            for i in range(b % self.num_parts):
                part_size[i] += 1

        return part_size


class RandomSampling(ActiveLearningAgent):
    """
    Choose b random nodes.
    """

    def __init__(self, data, model, seed, args):
        super(RandomSampling, self).__init__(data, model, seed, args)

    def query(self, b):
        indice = np.random.choice(
            np.where(self.data.train_mask == 0)[0], b, replace=False
        )
        return torch.tensor(indice)


class MaximumDegree(ActiveLearningAgent):
    """
    Choose b nodes with maximum degrees.
    """

    def __init__(self, data, model, seed, args):
        super(MaximumDegree, self).__init__(data, model, seed, args)

    def query(self, b):
        degree = self.data.adj.sum(dim=0)
        # degree = self.data.adj.cpu().sum(dim=0)
        degree[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(degree, k=b)
        return indices


class MaximumEntropy(ActiveLearningAgent):
    """
    Choose b nodes with maximum entropy.
    """

    def __init__(self, data, model, seed, args):
        super(MaximumEntropy, self).__init__(data, model, seed, args)

    def query(self, b):
        logits = self.clf(self.data.x, self.data.edge_index)
        entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
        entropy[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(entropy, k=b)
        return indices


class MaximumDensity(ActiveLearningAgent):
    """
    Choose b nodes with maximum density.
    K-Means clustering over the last hidden representation.
    """

    def __init__(self, data, model, seed, args):
        super(MaximumDensity, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get propagated nodes
        x_embed = self.transform('embed').cpu()

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x_embed)

        # Calculate density
        centers = kmeans.cluster_centers_
        label = kmeans.predict(x_embed)
        centers = centers[label]
        dist_map = np.linalg.norm(x_embed - centers, axis=1)
        density = 1 / (1 + dist_map)

        density[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(torch.tensor(density), k=b)

        return indices


class MaximumCentrality(ActiveLearningAgent):
    """
    Choose b nodes with maximum Pagerank score.
    """

    def __init__(self, data, model, seed, args):
        super(MaximumCentrality, self).__init__(data, model, seed, args)

    def query(self, b):
        nxg = nx.Graph(self.data.g.to_networkx())
        page = torch.tensor(list(pagerank(nxg).values()))
        page[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(page, k=b)
        return indices


class AGE(ActiveLearningAgent):
    """
    Selects nodes which maximizes a linear combination of three metrics:
    PageRank, uncertainty and density.
    """

    def __init__(self, data, model, seed, args):
        super(AGE, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get density
        x_embed = self.transform('embed').cpu()
        N = x_embed.shape[0]

        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x_embed)
        centers = kmeans.cluster_centers_
        label = kmeans.predict(x_embed)
        centers = centers[label]
        dist_map = np.linalg.norm(x_embed - centers, axis=1)
        density = torch.tensor(1 / (1 + dist_map), dtype=torch.float32, device=self.args.device)

        # Get entropy
        logits = self.clf(self.data.x, self.data.edge_index)
        entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)

        # Get centrality
        nxg = nx.Graph(self.data.g.to_networkx())
        page = torch.tensor(list(pagerank(nxg).values()), dtype=torch.float32, device=self.args.device)

        # Get percentile
        percentile = (torch.arange(N, dtype=torch.float32, device=self.args.device) / N)
        id_sorted = density.argsort(descending=False)
        density[id_sorted] = percentile
        id_sorted = entropy.argsort(descending=False)
        entropy[id_sorted] = percentile
        id_sorted = page.argsort(descending=False)
        page[id_sorted] = percentile

        # Get linear combination
        alpha, beta, gamma = self.data.params['age']
        age_score = alpha * entropy + beta * density + gamma * page
        age_score[np.where(self.data.train_mask != 0)[0]] = 0
        _, indices = torch.topk(age_score, k=b)
        return indices


class CoreSetGreedy(ActiveLearningAgent):
    """
    K-Center clustering over the last hidden representation and return centers.
    """

    def __init__(self, data, model, seed, args):
        super(CoreSetGreedy, self).__init__(data, model, seed, args)

    def query(self, b):

        embed = self.transform('embed').cpu()
        indices = list(np.where(self.data.train_mask != 0)[0])

        for i in range(b):
            dist = metrics.pairwise_distances(embed, embed[indices], metric='euclidean')
            min_distances = torch.min(torch.tensor(dist), dim=1)[0]
            new_index = min_distances.argmax()
            indices.append(int(new_index))
        return indices


class CoreSetMIP(ActiveLearningAgent):
    """
    K-Center clustering over the last hidden representation and return centers.
    Optimized by MIP.
    """

    def __init__(self, data, model, seed, args):
        super(CoreSetMIP, self).__init__(data, model, seed, args)

    def query(self, b):
        import gurobipy

        # Get distance matrix
        embed = self.transform('embed')
        dist_mat = embed.matmul(embed.t())
        sq = dist_mat.diagonal().reshape(self.data.num_nodes, 1)
        dist_mat = torch.sqrt(-dist_mat * 2 + sq + sq.t())

        # Perform greedy K-center
        mask = self.data.train_mask.copy()
        mat = dist_mat[~mask, :][:, mask]
        _, indices = mat.min(dim=1)[0].topk(k=b)
        indices = torch.arange(self.data.num_nodes)[~mask][indices]
        mask[indices] = True

        # Robust approximation
        opt = mat.min(dim=1)[0].max()
        ub = opt
        lb = opt / 2.0
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        flag = self.data.train_mask.copy()
        subset = np.where(flag == 0)[0].tolist()

        # Solve MIP for fac_loc
        x = {}
        y = {}
        z = {}
        n = self.data.num_nodes
        m = len(xx)

        model = gurobipy.Model("k-center")
        for i in range(n):
            z[i] = model.addVar(
                obj=1, ub=0.0, vtype="B", name="z_{}".format(i))

        for i in range(m):
            _x = xx[i]
            _y = yy[i]
            if _y not in y:
                if _y in subset:
                    y[_y] = model.addVar(
                        obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
                else:
                    y[_y] = model.addVar(
                        obj=0, vtype="B", name="y_{}".format(_y))
            x[_x, _y] = model.addVar(
                obj=0, vtype="B", name="x_{},{}".format(_x, _y))
        model.update()

        coef = [1 for j in range(n)]
        var = [y[j] for j in range(n)]
        model.addConstr(
            gurobipy.LinExpr(coef, var), "=", rhs=b + len(subset), name="k_center")

        for i in range(m):
            _x = xx[i]
            _y = yy[i]
            model.addConstr(
                x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))

        yyy = {}
        for v in range(m):
            _x = xx[v]
            _y = yy[v]
            if _x not in yyy:
                yyy[_x] = []
            if _y not in yyy[_x]:
                yyy[_x].append(_y)

        for _x in yyy:
            coef = []
            var = []
            for _y in yyy[_x]:
                coef.append(1)
                var.append(x[_x, _y])
            coef.append(1)
            var.append(z[_x])
            model.addConstr(
                gurobipy.LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))

        # Approximate
        delta = 1e-7
        sol_file = None
        while ub - lb > delta:
            cur_r = (ub + lb) / 2.0
            viol = np.where(dd > cur_r)
            new_max_d = torch.min(dd[dd >= cur_r])
            new_min_d = torch.max(dd[dd <= cur_r])
            for v in viol[0]:
                x[xx[v], yy[v]].UB = 0

            model.update()
            r = model.optimize()
            if model.getAttr(gurobipy.GRB.Attr.Status) == gurobipy.GRB.INFEASIBLE:
                failed = True
                print("Infeasible")
            elif sum([z[i].X for i in range(len(z))]) > 0:
                failed = True
                print("Failed")
            else:
                failed = False
            if failed:
                lb = max(cur_r, new_max_d)
                for v in viol[0]:
                    x[xx[v], yy[v]].UB = 1
            else:
                print("sol founded", cur_r, lb, ub)
                ub = min(cur_r, new_min_d)
                sol_file = "s_{}_solution_{}.sol".format(b, cur_r)
                model.write(sol_file)

        # Process results
        if sol_file is not None:
            results = open(sol_file).read().split('\n')
            results_nodes = filter(lambda x1: 'y' in x1,
                                   filter(lambda x1: '#' not in x1, results))
            string_to_id = lambda x1: (
                int(x1.split(' ')[0].split('_')[1]),
                int(x1.split(' ')[1]))
            result_node_ids = map(string_to_id, results_nodes)
            centers = []
            for node_result in result_node_ids:
                if node_result[1] > 0:
                    centers.append(node_result[0])
            return torch.tensor(centers)
        else:
            return None


class KmeansNaive(ActiveLearningAgent):
    """
    K-Means clustering over the feature and return centers.

    Feature Space: x
    Partition: No
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(KmeansNaive, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get propagated nodes
        x_prop = self.transform('none')

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x_prop.cpu())
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        for center in centers:
            dist_map = np.linalg.norm(x_prop.cpu() - center, axis=1)
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class KmeansEmbed(ActiveLearningAgent):
    """
    K-Means clustering over the last hidden representation and return centers.

    Feature Space: embedding
    Partition: No
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(KmeansEmbed, self).__init__(data, model, seed, args)

    def query(self, b):
        # Get propagated nodes
        x_prop = self.transform('embed')

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x_prop.cpu())
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        for center in centers:
            dist_map = np.linalg.norm(x_prop.cpu() - center, axis=1)
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class FeaturePropagation(ActiveLearningAgent):
    """
    K-Means (approximation for K-Medoids) clustering
    over the propagated features and return centers.

    Feature Space: prop
    Partition: No
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(FeaturePropagation, self).__init__(data, model, seed, args)

    def query(self, b):

        # Get propagated nodes
        x_prop = self.transform('prop')

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x_prop.cpu())
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        for center in centers:
            dist_map = np.linalg.norm(x_prop.cpu() - center, axis=1)
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class GraphPartition(ActiveLearningAgent):
    """
    K-Means clustering over the feature over each graph partition.

    Feature Space: x
    Partition: Yes
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(GraphPartition, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            self.num_centers = 1
        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Determine the number of partitions and number of centers
        part_size = self.get_n_cluster(b, partitions)

        # Get propagated nodes
        x_embed = self.transform('none')

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1)
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)


class GraphPartitionEmbed(ActiveLearningAgent):
    """
    K-Means clustering over the embedding over each graph partition.

    Feature Space: embedding
    Partition: Yes
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(GraphPartitionEmbed, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            self.num_centers = 1
        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Determine the number of partitions and number of centers
        part_size = self.get_n_cluster(b, partitions)

        # Get propagated nodes
        x_embed = self.transform('embed')

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1)
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)


class GraphPartitionProp(ActiveLearningAgent):
    """
    K-Means clustering over the embedding over each graph partition.

    Feature Space: prop
    Partition: Yes
    Avoid Known: No
    """

    def __init__(self, data, model, seed, args):
        super(GraphPartitionProp, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            self.num_centers = 1

        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Get propagated nodes
        x_embed = self.transform('prop')

        part_size = self.get_n_cluster(b, partitions)

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1)
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)


class FarFromKnown(ActiveLearningAgent):
    """
    K-Means clustering over the feature and avoid known nodes

    Feature Space: x
    Partition: No
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarFromKnown, self).__init__(data, model, seed, args)

    def query(self, b):

        # Get propagated nodes
        x = self.transform('none').cpu()

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x)
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        dist_to_center = np.ones(x.shape[0]) * np.infty

        for idx in indices:
            dist_to_center = np.minimum(dist_to_center, np.linalg.norm(x - x[idx], axis=1))

        for center in centers:
            dist_map = np.linalg.norm(x.cpu() - center, axis=1) - dist_to_center
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class FarEmbed(ActiveLearningAgent):
    """
    K-Means clustering over the embedding and avoid known nodes

    Feature Space: embedding
    Partition: No
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarEmbed, self).__init__(data, model, seed, args)

    def query(self, b):

        # Get propagated nodes
        x = self.transform('embed').cpu()

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x)
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        dist_to_center = np.ones(x.shape[0]) * np.infty

        for idx in indices:
            dist_to_center = np.minimum(dist_to_center, np.linalg.norm(x - x[idx], axis=1))

        for center in centers:
            dist_map = np.linalg.norm(x.cpu() - center, axis=1) - dist_to_center
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class FarProp(ActiveLearningAgent):
    """
    K-Means clustering over the propagated features and avoid known nodes

    Feature Space: embedding
    Partition: No
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarProp, self).__init__(data, model, seed, args)

    def query(self, b):

        # Get propagated nodes
        x = self.transform('prop').cpu()

        # Perform K-Means as approximation
        kmeans = KMeans(n_clusters=b, init='k-means++')
        kmeans.fit(x)
        centers = kmeans.cluster_centers_

        indices = list(np.where(self.data.train_mask != 0)[0])
        dist_to_center = np.ones(x.shape[0]) * np.infty

        for idx in indices:
            dist_to_center = np.minimum(dist_to_center, np.linalg.norm(x - x[idx], axis=1))

        for center in centers:
            dist_map = np.linalg.norm(x.cpu() - center, axis=1) - dist_to_center
            dist_map[indices] = np.infty
            idx = np.argmin(dist_map)
            indices.append(idx)
        return torch.tensor(indices)


class FarPartition(ActiveLearningAgent):
    """
    K-Means clustering over the feature on each partition and avoid known nodes

    Feature Space: x
    Partition: Yes
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarPartition, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            self.num_centers = 1
        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Determine the number of partitions and number of centers
        part_size = self.get_n_cluster(b, partitions)

        x_embed = self.transform('none')
        
        N = x_embed.shape[0]
        decay = 1

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        dist_to_center = np.ones(N) * np.infty
        for idx in indices:
            dist_to_center = np.minimum(
                dist_to_center, np.linalg.norm(x_embed.cpu() - x_embed[idx].cpu(), axis=1)
            )

        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]
            dist = dist_to_center[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1) - dist * decay
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])

        return indices


class FarPartitionEmbed(ActiveLearningAgent):
    """
    K-Means clustering over the embedding on each partition and avoid known nodes

    Feature Space: embedding
    Partition: Yes
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarPartitionEmbed, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            self.num_centers = 1
        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Determine the number of partitions and number of centers
        part_size = self.get_n_cluster(b, partitions)

        x_embed = self.transform('none')

        N = x_embed.shape[0]
        decay = 1

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        dist_to_center = np.ones(N) * np.infty
        for idx in indices:
            dist_to_center = np.minimum(dist_to_center, np.linalg.norm(x_embed.cpu() - x_embed[idx].cpu(), axis=1))

        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]
            dist = dist_to_center[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1) - dist * decay
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])

        return indices


class FarPartitionProp(ActiveLearningAgent):
    """
    K-Means clustering over the propagated embedding.

    Feature Space: prop
    Partition: Yes
    Avoid Known: Yes
    """

    def __init__(self, data, model, seed, args):
        super(FarPartitionProp, self).__init__(data, model, seed, args)

    def query(self, b):

        # Perform graph partition
        self.num_parts = int(np.ceil(b / self.num_centers))
        decay = 0
        if self.num_parts > self.data.max_part:
            self.num_parts = self.data.max_part
            decay = 1

        partitions = np.array(self.data.partitions[self.num_parts].cpu())

        # Get propagated nodes
        x_embed = self.transform('prop')
        part_size = self.get_n_cluster(b, partitions)

        # Perform K-Means as approximation
        indices = list(np.where(self.data.train_mask != 0)[0])
        for i in range(self.num_parts):
            part_id = np.where(partitions == i)[0]
            masked_id = [i for i, x in enumerate(part_id) if x in indices]
            x = x_embed[part_id]

            n_clusters = part_size[i]
            if n_clusters <= 0:
                continue

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans.fit(x.cpu())
            centers = kmeans.cluster_centers_

            for idx in indices:
                dist_to_center = np.ones(x_embed.shape[0]) * np.infty
                dist_to_center = np.minimum(dist_to_center, np.linalg.norm(x_embed.cpu() - x_embed[idx].cpu(), axis=1))
            dist = dist_to_center[part_id]

            for center in centers:
                dist_map = np.linalg.norm(x.cpu() - center, axis=1)
                dist_map -= dist * decay
                dist_map[masked_id] = np.infty
                idx = np.argmin(dist_map)
                masked_id.append(idx)
                indices.append(part_id[idx])
        return torch.tensor(indices)
