# coding: utf-8

import sys
import pandas as pd
import numpy as np
import networkx as nx

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
def build_dataset(num_files,show):
    rows = num_files*120052 # 120052 rows from each file
    cols = 100+10  # 10x10 adj list + 10 optimal labels
    data = np.zeros((rows,cols))
    row = 0
    processed = 0
    for idx in range(num_files):
        # get Y
        target_file_path = '{}opt_seq_n10_{}.g6.txt'.format(RESULT_DATA_PATH,idx)
        df = pd.read_csv(target_file_path, sep=';', dtype=int, header=None, skiprows=1,
                         usecols=list(range(1,11)))
        Y = df.values
        
        # get x and store it in data
        data_file_path = '{}n10_{}.g6'.format(G6_GRAPH_PATH,idx)
        G = nx.read_graph6(data_file_path)
        for i,graph in enumerate(G):
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

