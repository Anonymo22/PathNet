# PAGG: Path Aggregator for Graphs beyond Homophily

Graph neural networks (GNNs) have been intensively studied in various real-world tasks. However, the homophily assumption of GNNs' aggregation function limits their representation learning ability in heterophily graphs.
In this paper, we shed light onto the path level patterns in graphs that can explicitly reflect richer semantic meanings, especially the complex formation law of heteropily graphs.
We therefore propose a novel Path AGGregation model (PAGG) to learn diverse path information in a graph. Specifically, we first introduce a path sampler, which helps us sample a limited number of paths without bias from each node, and provides a theoretical guarantee for it. Then, we introduce a semantic distinguishable path encoder to learn the semantic meaning of different paths. Finally, we design a novel GNN model to integrate the above components.
The experimental results demonstrate that our models obtain an improvement of +8.65\% on average in terms of accuracy for node classification task in heterophily graphs, while also showing competitive results in homophily graphs.

## Path Sampler (preprocessing)
This part can be done before training.
To generate the paths for dataset *data_name* (*e.g.* cora). In *gen.cpp*, we change the names of the input and output files to *data_name*, then compile and run *gen.cpp*. The program will generate a file containing all paths.

Compile and run *gen.cpp*  for normal datasets

```shell
g++ gen.cpp -o gen -g -Wall -O2 -mcmodel=medium
./gen
```

If a graph contains too many nodes, the file generated by the above method will be too large and occupy a large running memory. So we provide *gen_epoch.cpp* to generate corresponding paths for each epoch.

Compile and run *gen_epoch.cpp* for large datasets

```shell
g++ gen_epoch.cpp -o gen_epoch -g -Wall -O2 -mcmodel=medium
./gen_epoch
```
## Main model

### Files
'dataset.json' contains dataset splits