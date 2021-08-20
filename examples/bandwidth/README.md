# Bandwidth minimization

The bandwidth minimization problem consists in relabeling the Nodes in a Graph, in a way that when this graph is represented in a form of a adjacency matrix, the non-zero values are near the main diagonal. This folder contains files just like the 'exact_bandwidth' folder, but executes in a minor scale, processing graphs of 3, 5 until 7 vertices or nodes, this is in order to help us understanding the process. The functions developed by Augusto Fonseca are pretty much the same, with some changes, as passing a CLI number argument for example.

The process is the same as in the exact_bandwidth folder:

- **The first step requires a Ubuntu/Linux machine**

1 - Execute **opt_band_example.py** (with number 3, 5 or 7 as required CLI argument, exact_bandwidth is set up for 10 nodes) to calculate the optimal band of the Graph.

2 - Execute **build_dataset_v2_example.py** to build the opt_band_n_nodes_graph_example.npy dataset. This is the Graphs in the form of adjacency matrix

Now, since the dataset is available, we can run the Neural Network.

3 - Execute **neural_network_v2_example.ipynb** jupyter notebook to run the neural network. This will make use of the Features and Labels in the "opt_band_10_nodes_graph.npy" data since it is a Supervised learning.

## More detailed file descriptions

## opt_band_example.py

This file will generate txt (a text represented Graph) files in the "txt_graphs_files_example" folder, this is required since a C software, CPLEX Optimizer (MBandwidth), uses these files to return the optimal sequence of a Graph. The optimal exact sequences for all non-isomorphic graphs of, for example, 5 nodes, is stored in the optimalSequence_n5_g6.txt file, under the "opt_results/n5_blocks".

To be executed requires the number of nodes for all non-isomorphic graphs of that number and a flag to start a clean execution deleting old files and generating new ones.

Usage example, for 5 nodes:

```shell
python opt_band_example.py 5 Y
```

## build_dataset_v2_example.py

## neural_network_v2_example.ipynb
