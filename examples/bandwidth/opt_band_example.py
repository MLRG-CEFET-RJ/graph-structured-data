import sys
import os
import subprocess
import networkx as nx
import shutil
import numpy as np
from networkx.utils import reverse_cuthill_mckee_ordering

TXT_GRAPHS_FILES_EXAMPLE_PATH = './txt_graphs_files_example/'
M_BANDWIDTH_PATH = '../../exact_bandwidth/opt_band/bin/MBandwidth'
PATH_TO_GRAPHS_DATASETS = '../../datasets/graph6'
BUFFER = 300

def writeGraphAsTextFile(graph, idx):
    # Write a '.txt' file for each non-isomorphic graph of n nodes
    # i.e. each '0.txt, 1.txt...' file is a Graph of n nodes.
    edges = graph.edges()
    numberOfNodes = len(graph.nodes())
    numberOfEdges = len(edges)
    text = f'{numberOfNodes} {numberOfEdges}\n'
    # besides storing the edges relations in 'text', we increment the value by 1, being 1-based indexing
    for edge in edges:
        text += f'{edge[0]+1} {edge[1]+1}\n'

    if not os.path.exists(TXT_GRAPHS_FILES_EXAMPLE_PATH):
        os.mkdir(TXT_GRAPHS_FILES_EXAMPLE_PATH)
    filename = f'{TXT_GRAPHS_FILES_EXAMPLE_PATH}/{idx}.txt'

    with open(filename, 'w') as writer:
        writer.write(text)
    # The written text file consists in the following:
    # 'Number of Nodes' 'Number of Edges (first line)
    # 'nodeX' 'nodeY' (second line and so on, meaning an edge between these two nodes)
    # Note that if you pass 3 as CLI argument it will create 4 ".txt" files,
    # which are the four non-isomorphic graphs of 3 nodes
    # This text files are required because the IBM software uses text files as input

def get_result(stdout, num_nodes):
    # the following line means, for a graph of 3 nodes, 
    # the last 3 lines in the MBandWidth output ([b'0 0\n', b'1 2\n', b'2 1\n'])
    # node 0 continue being 0, node 1 gets label 2, node 2 gets label 1
    result_lines = stdout[-(num_nodes):]
    result = list()
    for line in result_lines:
        result.append(line[2:-1].decode()) # removing old nodes and '\n'
    return result

def saveAllGraphsOptimalBandsInOneTextFile(numberGraphs, result_file, numberNodes, block):
    path = f'./opt_results/n{numberNodes}_blocks/'
    if not os.path.exists(path):
        os.mkdir('./opt_results')
        os.mkdir(path)
    path += result_file

    with open(path, 'w') as writer:
        header = ' '*150 + '\n' # header of the file will be filled later
        writer.write(header)
        records = ''
        for i in range(numberGraphs):
            if i % 300 == 0:
                print(f"Executing optimal sequence for {numberGraphs} graphs under './txt_grahps_files_example' folder. Current Graph - {i} of block {block}")

            file_path = f'{TXT_GRAPHS_FILES_EXAMPLE_PATH}{i}.txt'

            # Note that, in order to proper execute the next line, the script must be 
            # running in a Ubuntu/Linux operating system. 
            # I've executed on a Windows 10 machine and the following error happens:
            # [WinError 193] %1 is not a valid Win32 application
            cp = subprocess.Popen([M_BANDWIDTH_PATH, file_path, '3600', '1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            lines = cp.stdout.readlines()

            # Uncomment the following lines to see the output of the MBandWidth software
            # note how the 1.txt for 3 nodes graph, has a bandwidth of 2 and it's optimal bandwidth is 1
            # same note for 2.txt, for a 3 nodes graph
            # print("LINESS")
            # print(type(lines)) # list of bytes
            # print(f"Processing non-isomorphic graph ({numberNodes} nodes) => {file_path} ")
            # for j in range(len(lines)):
            #     print(lines[j])
            # print("END LINES")

            opt_sequence = get_result(lines, numberNodes)
            opt_sequence = ';'.join(opt_sequence)
            # print(f"{str(i)}.txt - {opt_sequence}")
            line = f'{i};{opt_sequence}\n'
            records += line
            if i % BUFFER == 0:
                writer.write(records)
                records = ''
        # last records
        if records:
            writer.write(records)

def load_opt_seq(file_name, numberNodes):
    path = f'./opt_results/n{numberNodes}_blocks/{file_name}'
    with open(path,'r') as reader:
        lines = reader.readlines()
    optimal_sequence_dict = dict()
    for line in lines[1:]: # removing header line (150 empty blank spaces, to be filled later)
        sequence = list(map(int, line[:-1].split(';'))) # remove final \n and map values to integers 
        optimal_sequence_dict[sequence[0]] = sequence[1:] # v[0] == idGraph, v[1:] == optimal sequence
    return optimal_sequence_dict

def clean(cleanArg):
    if os.path.exists(r'./opt_results') or os.path.exists(r'./txt_graphs_files_example'):
        if cleanArg != 'Y' and cleanArg != 'y':
            return
        shutil.rmtree(r"./opt_results")
        shutil.rmtree(r"./txt_graphs_files_example")

def get_bandwidth_nodelist_adjacency_rcm(Graph):
    rcm = list(reverse_cuthill_mckee_ordering(Graph))
    A = nx.adjacency_matrix(Graph, nodelist=rcm)
    L = nx.laplacian_matrix(nx.Graph(A))
    x, y = np.nonzero(L)
    return (x-y).max()


def get_cp_fixed(cp_nodelist_order):
    order_fixed = [0 for _ in range(len(cp_nodelist_order))]
    for idx, element in enumerate(cp_nodelist_order):
        order_fixed[element] = idx
    return order_fixed

def get_bandwidth_nodelist_adjacency_cp(Graph, id, optimal_sequence_dict):
    cp = optimal_sequence_dict[id]
    A = nx.adjacency_matrix(Graph, nodelist=get_cp_fixed(cp))
    L = nx.laplacian_matrix(nx.Graph(A))
    x, y = np.nonzero(L)
    return (x-y).max()

def test_result(Graphs, optimal_sequence_dict):
    # larger = [(idx, exact_band, heuristic_band)]
    larger = list()
    same = 0
    smaller = 0

    start = 1
    
    for i in range(start,len(Graphs)):
        heuristic_band = get_bandwidth_nodelist_adjacency_rcm(Graphs[i])
        cp_band = get_bandwidth_nodelist_adjacency_cp(Graphs[i], i, optimal_sequence_dict)

        if cp_band < heuristic_band:
            smaller += 1
        elif cp_band == heuristic_band:
            same += 1
        else:
            larger.append((i, cp_band, heuristic_band))
    
    return larger,same, smaller

def getNumberOfG6Files(numberNodes):
    # each g6 file contains a list of graphs.
    # thus, each file is considered a block of graphs
    # note that, for 5, 7 and 9 there is only 1 block of graphs 
    numberOfBlocks = os.listdir(f'{PATH_TO_GRAPHS_DATASETS}/n{numberNodes}_blocks')
    return len(numberOfBlocks)

def cleanOnlyTextGraphs():
    # clean text files of a block, to fill it up with the next block to be executed
    shutil.rmtree(r"./txt_graphs_files_example")

def writeOptimalSequenceTextFileForBlock(block, numberNodes):
    # get a block of graphs to write the optimal sequences file for that block, under "opt_results" folder
    # the "optimal sequences" file contains all optimal sequences for each graph in that block 
    Graphs = nx.read_graph6(f'{PATH_TO_GRAPHS_DATASETS}/n{numberNodes}_blocks/n{numberNodes}_{block}.g6')
    print(f"There are {len(Graphs)} non-isomorphic graphs of {numberNodes} nodes in the block {block} (.g6 file)")
    for i in range(len(Graphs)):
        writeGraphAsTextFile(Graphs[i], i)

    result_file = f'optimalSequences_n{numberNodes}_{block}.g6.txt'
    saveAllGraphsOptimalBandsInOneTextFile(len(Graphs), result_file, numberNodes, block)

    optimal_sequence_dict = load_opt_seq(result_file, numberNodes)

    larger,same,smaller = test_result(Graphs, optimal_sequence_dict)

    # Write tests results in the 150 blanks spaces that were reserved
    arr = [len(larger),same,smaller]
    s = ';'.join(list(map(str, arr)))
    path = f'./opt_results/n{numberNodes}_blocks/{result_file}'
    with open(path, 'r+') as file:
        file.seek(0) # move a cursor writer (or reader) to position 0
        file.write(s)

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3 or sys.argv[1] not in ['3', '5', '7', '9', '10'] or sys.argv[2] == ' ':
            raise ValueError("Number of nodes in the Graphs and clean flag required as arguments\nCLI usage:\npython opt_band_example.py [3|5|7|9|10] [Y|N]\n# '5' and 'Y' recommended args")
        file, numberNodes, cleanArg = sys.argv
        numberNodes = int(numberNodes)
        # clean the text files and optimal sequence(s) file(s) to make a clean run
        clean(cleanArg)
        # Read graphs
        print(f'Processing all non-isomorphic Graphs of {numberNodes} nodes...')
        numberG6Files = getNumberOfG6Files(numberNodes)
        for block in range(numberG6Files):
            if numberG6Files > 1:
                cleanOnlyTextGraphs()
                # last execution will maintain the last block
            writeOptimalSequenceTextFileForBlock(block, numberNodes)
    except Exception as e:
        print('Error:')
        print(e)