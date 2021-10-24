# PAGG: Path Aggregator for Graphs beyond Homophily

Graph neural networks (GNNs) have been intensively studied in various real-world tasks. However, the homophily assumption of GNNs' aggregation function limits their representation learning ability in heterophily graphs.
In this paper, we shed light onto the path level patterns in graphs that can explicitly reflect richer semantic meanings, especially the complex formation law of heteropily graphs.
We therefore propose a novel Path AGGregation model (PAGG) to learn diverse path information in a graph. Specifically, we first introduce a path sampler, which helps us sample a limited number of paths without bias from each node, and provides a theoretical guarantee for it. Then, we introduce a semantic distinguishable path encoder to learn the semantic meaning of different paths. Finally, we design a novel GNN model to integrate the above components.
The experimental results demonstrate that our models obtain an improvement of +8.65\% on average in terms of accuracy for node classification task in heterophily graphs, while also showing competitive results in homophily graphs.

# Note

Code is being sorted out, coming in 9 hours...