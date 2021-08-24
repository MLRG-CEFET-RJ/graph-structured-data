import sys
import pandas as pd
import numpy as np
import networkx as nx
import os
from opt_band_example import get_cp_fixed, PATH_TO_GRAPHS_DATASETS

def get_optimal_bandwidth(G, nodelist):
    if len(G.edges) == 0:
        return 0
    L = nx.laplacian_matrix(G,nodelist=nodelist)
    x,y = np.nonzero(L)
    return (x-y).max()

def buildCSVDataset(data, numberNodes):
    digits = (numberNodes * numberNodes - numberNodes) // 2
    bandwidthValue = 1
    labels = numberNodes
    columns = []

    for i in range(digits):
        columns.append(f'xDigit_{i}')
    columns.append("opt_band")
    for j in range(labels):
        columns.append(f"yLabel_{j}")

    df = pd.DataFrame(data, columns=tuple(columns))
    csv_text = df.to_csv(index=False, line_terminator='\n')
    with open(f'../../datasets/examples/opt_band_{numberNodes}_nodes_graph.csv', 'w') as writer:
        writer.write(csv_text)

def build_dataset(*args):
    numberBlocks, numberNodes, numberTotalSequences, target, verbose = args

    rows = numberTotalSequences
    numberDigitsAdjcencyMatrix = (numberNodes * numberNodes - numberNodes) // 2
    # cols = 45 + 1 + 10  
    # 10x10 upper triangle not optimal adjcency list + optimal_band (value) + 10 optimal labels (nodelist)
    # We are handling with symmetric adjcency lists (get the upper triangle from the main diagonal)
    cols = numberDigitsAdjcencyMatrix + 1 + numberNodes
    data = np.zeros((rows,cols))
    row = 0
    for block in range(numberBlocks):
        optimalSequence_i_file = f'./opt_results/n{numberNodes}_blocks/optimalSequences_n{numberNodes}_{block}.g6.txt'
        df = pd.read_csv(optimalSequence_i_file, sep=';', dtype=int, header=None, skiprows=1, usecols=list(range(1, numberNodes + 1)))
        optimalSequences = df.values
        optimalSequences = list(map(get_cp_fixed, optimalSequences))
        optimalSequences = np.array(optimalSequences)
        # optimalSequences is a matrix, idx 0 (row 0) is graph 0, 
        # index 0 contains a list that represents its optimal sequence nodelist.
        # Data from the optimalSequence_n{NUMBER_NODES}_g6.txt
        # of course that another approach would be use load_opt_seq function, 
        # but rather than returning a dict like load_opt_seq, this time we got an array

        # each row holds the upper triangle flattened, optimal bandwidth and optimal nodelist as columns
        # this will be the dataset to be passed into the neural network, stored as a ".npy" matrix
        Graphs = nx.read_graph6(f'{PATH_TO_GRAPHS_DATASETS}/n{numberNodes}_blocks/n{numberNodes}_{block}.g6')
        for i, graph in enumerate(Graphs):
            opt_band = get_optimal_bandwidth(graph, optimalSequences[i])
            floatAdjMatrix = nx.to_numpy_array(graph)
            # "nx.to_numpy_array" is the same as "nx.adjacency matrix", but later we'll
            # use pytorch, a neural network works better with floats, since we have lot of 'wx + b' operations
            upperTriangleFlatten = np.array([floatAdjMatrix[row][column] for row in range(numberNodes - 1) for column in range(row + 1, numberNodes)])
            data[row] = np.concatenate((np.array(upperTriangleFlatten, copy=True), np.array([opt_band]), optimalSequences[i]))
            row += 1
        if verbose and block % (numberTotalSequences // 4) == 0:
            print(f'{block + 1} blocks processed, total of {len(Graphs)} optimal sequences in the block just executed.')
    # np.save(target, data)
    buildCSVDataset(data, numberNodes)

def getNumberOptimalSequenceFiles(numberNodes):
    arr = os.listdir(f'./opt_results/n{numberNodes}_blocks')
    if len(arr) == 0:
        # there's no optimal sequence file(s), because opt_band_example wasn't executed previously
        raise FileNotFoundError(f"Optimal sequence(s) file(s) for n{numberNodes}_blocks required to build the dataset.\nExecute 'python opt_band_example.py {numberNodes} Y' first.")
    return len(arr)

def get_number_of_total_sequences(numberNodes, numberBlocks):
    # each block have generated a "optimalSequences_n{n}_{0}.g6.txt" that contains a amount of optimal sequences for its block
    # for "optimalSequences_n10_0.g6.txt" there are 120052 optimal sequences,
    # for "optimalSequences_n10_99.g6.txt" there are 120020 optimal sequences
    Graphs = nx.read_graph6(f'{PATH_TO_GRAPHS_DATASETS}/n{numberNodes}_blocks/n{numberNodes}_0.g6')
    total = (numberBlocks - 1) * len(Graphs)
    # numberBlocks - 1, because until the last, all blocks has the same number of graphs, thus same number of sequences,
    # if only 1 block, total = 0
    Graphs = nx.read_graph6(f'{PATH_TO_GRAPHS_DATASETS}/n{numberNodes}_blocks/n{numberNodes}_{numberBlocks - 1}.g6') # numberBlocks - 1, it's 0 indexed
    total += len(Graphs)
    return total

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3 or sys.argv[1] not in ['3', '5', '7', '9', '10'] or sys.argv[2] == ' ':
            raise ValueError("Number of nodes in the Graphs and verbose flag required as arguments\nCLI usage:\npython build_dataset_v2_example.py [3|5|7] [0|1]\n# '5' and '0' recommended args")
        file, numberNodes, verbose = sys.argv
        numberNodes = int(numberNodes)
        verbose = int(verbose)
        
        numberBlocks = getNumberOptimalSequenceFiles(numberNodes)
        numberTotalSequences = get_number_of_total_sequences(numberNodes, numberBlocks)
        target = f'../../datasets/examples/opt_band_{numberNodes}_nodes_graph.csv'
        print(f"Processing {numberBlocks} optimal sequence files, total of {numberTotalSequences} sequences...")
        build_dataset(numberBlocks, numberNodes, numberTotalSequences, target, verbose)

        
        # dataset = np.load(target)
        print(f'Dataset created in path {target}. head: \n{pd.read_csv(target).head()}')
    except Exception as e:
        print('Error:\n')
        print(e)
        exception_traceback = sys.exc_info()[2]
        line_number = exception_traceback.tb_lineno
        print("Error Line: ", line_number)