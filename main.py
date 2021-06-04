from __future__ import division
from __future__ import print_function

import argparse
import random

from models import GCN, GAT
from query import *
from dataset import load_data


# Training settings
parser = argparse.ArgumentParser()

# General configs
parser.add_argument(
    "--retrain", type=bool, default=True)
parser.add_argument(
    "--baseline", type=str, default='random')
parser.add_argument(
    "--dataset", default='cora')
parser.add_argument(
    "--model", default="gcn")

# Training parameters
parser.add_argument(
    "--init", type=float, default=5, help="Number of initially labelled nodes.")
parser.add_argument(
    "--budget", type=list, default=[25, 30, 35, 40, 45, 55, 65, 75, 85, 95, 115, 135, 155, 175, 195],
    help="Number of rounds to run the agent.")
parser.add_argument(
    "--rounds", type=int, default=1, help="Number of rounds to run the agent.")
parser.add_argument(
    "--num_centers", type=int, default=2, help="Default number of centers per partition.")
parser.add_argument(
    "--epochs", type=int, default=300, help="Number of epochs to train.")
parser.add_argument(
    "--seed", type=int, default=10, help="Number of random seeds.")
parser.add_argument(
    "--verbose", type=int, default=0, help="Verbose.")
parser.add_argument(
    "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Common hyper-parameters
parser.add_argument(
    "--lr", type=float, default=0.01, help="Initial learning rate.")
parser.add_argument(
    "--weight_decay", type=float, default=5e-4,
    help="Weight decay (L2 loss on parameters).")
parser.add_argument(
    "--hidden", type=int, default=16, help="Number of hidden units.")
parser.add_argument(
    "--dropout", type=float, default=0,
    help="Dropout rate (1 - keep probability).")
parser.add_argument(
    "--activation", default="relu")

# GAT hyper-parameters
parser.add_argument(
    "--num_heads", type=int, default=4, help="Number of heads.")

args, _ = parser.parse_known_args()

# Load dataset
data = load_data(name=args.dataset).to(args.device)

# Choose model
model_args = {
    "num_features": data.num_features,
    "num_classes": data.num_classes,
    "hidden_size": args.hidden,
    "dropout": args.dropout,
    "activation": args.activation
}

# Set seeds
for budget in args.budget:
    budget = int(budget)
    for seed in range(args.seed):
        if args.verbose == 1:
            print('Seed {:03d}:'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if args.model == "gat":
            model_args["num_heads"] = args.num_heads
            model_args["hidden_size"] = int(args.hidden / args.num_heads)
            model = GAT(**model_args)
        else:
            model = GCN(**model_args)

        model = model.to(args.device)

        # Choose agent
        # Naive Methods
        if args.baseline == "random":
            agent = RandomSampling(data, model, seed, args)
        elif args.baseline == "degree":
            agent = MaximumDegree(data, model, seed, args)
        elif args.baseline == "entropy":
            agent = MaximumEntropy(data, model, seed, args)
        elif args.baseline == "density":
            agent = MaximumDensity(data, model, seed, args)
        elif args.baseline == "centrality":
            agent = MaximumCentrality(data, model, seed, args)
        elif args.baseline == "rwcs":
            agent = RWCS(data, model, seed, args)

        # Coreset
        elif args.baseline == "coreset":
            agent = CoreSetGreedy(data, model, seed, args)

        # Combined Methods
        elif args.baseline == "age":
            agent = AGE(data, model, seed, args)

        # Ablation Study
        elif args.baseline == "kmeans":
            agent = KmeansNaive(data, model, seed, args)
        elif args.baseline == "kmeans-embed":
            agent = KmeansEmbed(data, model, seed, args)
        elif args.baseline == "feature-propagation":
            agent = FeaturePropagation(data, model, seed, args)

        elif args.baseline == "graph-partition":
            agent = GraphPartition(data, model, seed, args)
        elif args.baseline == "graph-partition-embed":
            agent = GraphPartitionEmbed(data, model, seed, args)
        elif args.baseline == "graph-partition-prop":
            agent = GraphPartitionProp(data, model, seed, args)

        # Far
        elif args.baseline == "far-partition-prop":
            agent = FarPartitionProp(data, model, seed, args)
        else:
            agent = None
            exit(-1)

        # Initialization
        training_mask = np.zeros(data.num_nodes, dtype=bool)
        initial_mask = np.arange(data.num_nodes)
        np.random.shuffle(initial_mask)
        training_mask[initial_mask[:args.init]] = True

        training_mask = torch.tensor(training_mask)
        agent.update(training_mask)
        agent.train()

        if args.verbose > 0:
            print('Round {:03d}: Labelled: {:d}, Prediction macro-f1 score {:.4f}'
                  .format(0, args.init, agent.predict()))

        # Experiment
        for rd in range(1, args.rounds+1):
            # query
            indices = agent.query(budget)
            training_mask[indices] = True

            # update
            agent.update(training_mask)
            agent.train()

            # predict
            f1 = agent.predict()
            labelled = len(np.where(agent.data.train_mask != 0)[0])

            if args.verbose > 0:
                print('Round {:03d}: # Labelled nodes: {:d}, Prediction macro-f1 score {:.4f}'
                      .format(rd, labelled, f1))
            else:
                print("{},{},{},{},{},{}"
                      .format(args.model, args.baseline,
                              args.dataset, seed,
                              labelled, f1))
