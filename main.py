from __future__ import division
from __future__ import print_function

import argparse
import random
from timeit import default_timer as timer

from models import GCN, GAT, SAGE
from query import *
from dataset import load_data


def run(args):

    # Load dataset
    data = load_data(name=args.dataset,
                     read=True, save=False).to(args.device)

    for gnn in args.model:
        for baseline in args.baselines:
            for budget in args.budget:
                budget = int(budget)
                seed = int(args.seed)

                # Set seeds
                if args.verbose == 1:
                    print('Seed {:03d}:'.format(seed))
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Choose model
                model_args = {
                    "in_channels": data.num_features,
                    "out_channels": data.num_classes,
                    "hidden_channels": args.hidden,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "activation": args.activation,
                    "batchnorm": args.batchnorm
                }

                # Initialize models
                if gnn == "gat":
                    model_args["num_heads"] = args.num_heads
                    model_args["hidden_channels"] = int(args.hidden / args.num_heads)
                    model = GAT(**model_args)
                elif gnn == "gcn":
                    model = GCN(**model_args)
                elif gnn == "sage":
                    model = SAGE(**model_args)
                else:
                    raise NotImplemented

                model = model.to(args.device)

                # General-Purpose Methods
                if baseline == "random":
                    agent = Random(data, model, seed, args)
                elif baseline == "density":
                    agent = Density(data, model, seed, args)
                elif baseline == "uncertainty":
                    agent = Uncertainty(data, model, seed, args)
                elif baseline == "coreset":
                    agent = CoreSetGreedy(data, model, seed, args)

                # Graph-specific Methods
                elif baseline == "degree":
                    agent = Degree(data, model, seed, args)
                elif baseline == "pagerank":
                    agent = PageRank(data, model, seed, args)
                elif baseline == "age":
                    agent = AGE(data, model, seed, args)
                elif baseline == "featprop":
                    agent = ClusterBased(data, model, seed, args,
                                         representation='aggregation',
                                         encoder='gcn')

                # Our Methods
                elif baseline == "graphpart":
                    agent = PartitionBased(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=0)
                elif baseline == "graphpartfar":
                    agent = PartitionBased(data, model, seed, args,
                                           representation='aggregation',
                                           encoder='gcn',
                                           compensation=1)

                # Ablation Studies
                elif 'part' in baseline:
                    agent = PartitionBased(data, model, seed, args,
                                           representation=args.representation,
                                           compensation=0)
                else:
                    agent = ClusterBased(data, model, seed, args,
                                         representation=args.representation)

                # Initialization
                training_mask = np.zeros(data.num_nodes, dtype=bool)
                initial_mask = np.arange(data.num_nodes)
                np.random.shuffle(initial_mask)
                init = args.init
                if baseline in ['density', 'uncertainty', 'coreset', 'age']:
                    init = budget // 3
                training_mask[initial_mask[:init]] = True

                training_mask = torch.tensor(training_mask)
                agent.update(training_mask)
                agent.train()

                if args.verbose > 0:
                    print('Round {:03d}: Labelled: {:d}, Prediction macro-f1 score {:.4f}'
                          .format(0, init, agent.evaluate()))

                # Experiment
                for rd in range(1, args.rounds + 1):
                
                    # Query
                    start = timer()
                    indices = agent.query(budget - init)
                    end = timer()
                    print('Total Query Runtime [s]:', end - start)

                    # Update
                    training_mask[indices] = True
                    agent.update(training_mask)

                    # Training
                    agent.train()

                    # Evaluate
                    f1, acc = agent.evaluate()
                    labelled = len(np.where(agent.data.train_mask != 0)[0])

                    if args.verbose > 0:
                        print('Round {:03d}: # Labelled nodes: {:d}, Prediction macro-f1 score {:.4f}'
                              .format(rd, labelled, f1))
                    else:
                        print("{},{},{},{},{},{},{}"
                              .format(gnn, baseline,
                                      args.dataset, seed,
                                      labelled, f1, acc))


if __name__ == '__main__':

    datasets = ['cora']
    gnns = ['gcn', 'sage', 'gat']
    budgets = [20, 40, 80]
    baselines = ['graphpart', 'graphpartfar']

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbose: 0, 1 or 2")
    parser.add_argument(
        "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # General configs
    parser.add_argument(
        "--baselines", type=list, default=baselines)
    parser.add_argument(
        "--model", default=gnns)
    parser.add_argument(
        "--dataset", default='cora')

    # Active Learning parameters
    parser.add_argument(
        "--budget", type=list, default=budgets,
        help="Number of rounds to run the agent.")
    parser.add_argument(
        "--retrain", type=bool, default=True)
    parser.add_argument(
        "--num_centers", type=int, default=1)
    parser.add_argument(
        "--representation", type=str, default='features')
    parser.add_argument(
        "--compensation", type=float, default=1.0)
    parser.add_argument(
        "--init", type=float, default=0, help="Number of initially labelled nodes.")
    parser.add_argument(
        "--rounds", type=int, default=1, help="Number of rounds to run the agent.")
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train.")
    parser.add_argument(
        "--steps", type=int, default=4, help="Number of steps of random walk.")

    # GNN parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="Number of random seeds.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4,
        help="Weight decay (L2 loss on parameters).")
    parser.add_argument(
        "--hidden", type=int, default=16, help="Number of hidden units.")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument(
        "--dropout", type=float, default=0,
        help="Dropout rate (1 - keep probability).")
    parser.add_argument(
        "--batchnorm", type=bool, default=False,
        help="Perform batch normalization")
    parser.add_argument(
        "--activation", default="relu")

    # GAT hyper-parameters
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of heads.")

    args, _ = parser.parse_known_args()

    for dataset in datasets:
        args.dataset = dataset
        run(args)
