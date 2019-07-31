# coding: utf-8

import sys
import pandas as pd
import numpy as np
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering

# user parameters
if len(sys.argv) != 3:
    print('USE {} [(int) NUM_FILES] [(int) SHOW]'.format(sys.argv[0]))
    quit()
else:
    num_files = int(sys.argv[1])
    show = int(sys.argv[2])

# paths
G6_GRAPH_PATH = '../datasets/graph6/n10_blocks/'
RESULT_DATA_PATH = 'opt_results/n10_blocks/'
TARGET = '../datasets/opt_band_10_nodes_graph.npy'

# functions
def get_bandwidth(G, nodelist):
    if len(G.edges) == 0:
        return 0
    L = nx.laplacian_matrix(G,nodelist=nodelist)
    x,y = np.nonzero(L)
    return (x-y).max()


def get_cp_fixed(cp_order):
    """
    example:
        [5,9,0,4,2,6,8,7,1,3]
        [0,1,2,3,4,5,6,7,8,9]
        [2,8,4,9,3,0,5,7,6,1]
    """
    order_fixed = list(range(10))
    count = 0
    for elem in cp_order:
        order_fixed[elem] = count
        count += 1
    return order_fixed


def build_dataset(num_files,show):
    rows = num_files*120052 # 120052 rows from each file
    cols = 100 + 1 + 10  # 10x10 adj list + optimal_band + 10 optimal labels
    data = np.zeros((rows,cols))
    row = 0
    processed = 0
    for idx in range(num_files):
        # get Y
        target_file_path = '{}opt_seq_n10_{}.g6.txt'.format(RESULT_DATA_PATH,idx)
        df = pd.read_csv(target_file_path, sep=';', dtype=int, header=None, skiprows=1,
                         usecols=list(range(1,11)))
        Y = df.values
        for i,permutation in enumerate(Y):
            p = get_cp_fixed(permutation)
            Y[i] = p.copy()

        
        # get x and store it in data
        data_file_path = '{}n10_{}.g6'.format(G6_GRAPH_PATH,idx)
        G = nx.read_graph6(data_file_path)
        for i,graph in enumerate(G):
            opt_band = get_bandwidth(graph,Y[i])
            x = nx.to_numpy_array(graph).ravel()
            data[row] = np.concatenate((x,Y[i]))
            row += 1
        if idx % show == 0:
            processed += show
            print('{} files processed.'.format(processed))
            
    np.save(TARGET, data)

# build dataset
build_dataset(num_files,show)

# test
dataset = np.load(TARGET)
print('Dataset created in path {} with shape {}'.format(TARGET,dataset.shape))

