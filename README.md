# GraphPart

This repository provides a PyTorch implementation for the Graph-Partition-based active learning framework and baselines for GNNs as described in the following paper:

[Partition-Based Active Learning for Graph Neural Networks](https://arxiv.org/abs/2201.09391)

## Requirements

See `setup.sh`.

## Run the Code

```bash
python main.py --baselines graphpart --model gcn --dataset cora --seed 0
```

