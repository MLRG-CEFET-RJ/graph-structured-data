#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import shutil
import subprocess

import networkx as nx
import numpy as np

from networkx.utils import reverse_cuthill_mckee_ordering


# param defined by user
initial_index_file = int(sys.argv[1]) if len(sys.argv) > 1 else 0


G6_DATA_PATH = '../datasets/graph6/n10_blocks/'
G6_DATA_NUM_BLOCKS = len([name for name in os.listdir(G6_DATA_PATH) if os.path.isfile(os.path.join(G6_DATA_PATH, name))])
GRAPH_DATA_PATH = 'txt_files/'
RESULT_DATA_PATH = 'opt_results/n10_blocks/'
BUFFER = 300
NUM_NODES = 10


def write_txt_file(graph, idx):
    E = graph.edges()
    txt = '{} {}\n'.format(len(graph.nodes()), len(graph.edges()))
    for e in E:
        txt += '{} {}\n'.format(e[0]+1,e[1]+1)

    path = '{}{}.txt'.format(GRAPH_DATA_PATH, idx)
    with open(path,'w') as f:
        f.write(txt)


def get_result(stdout, num_nodes):
    result_lines = stdout[-(num_nodes):]
    result = list()
    for line in result_lines:
        result.append(line[2:-1].decode()) # removing old nodes and '\n'
    return result


def load_graphs(file_name):
    return nx.read_graph6('{}{}'.format(G6_DATA_PATH, file_name))


def get_bandwidth(method_name, G, nodelist=None):
    if not nodelist:
        nodelist = sorted(G.nodes())
    elif method_name == 'CP':
        nodelist = get_cp_fixed(nodelist)
    L = nx.laplacian_matrix(G,nodelist=nodelist)
    x,y = np.nonzero(L)
    return (x-y).max()


def get_cp_fixed(cp_order):
    """
    example:
        [5,9,0,4,2,6,8,7,1,3] -> cp_order == new labels
        [0,1,2,3,4,5,6,7,8,9] -> old labels
        [2,8,4,9,3,0,5,7,6,1] -> order in rcm mode
    """
    order_fixed = list(range(NUM_NODES))
    count = 0
    for elem in cp_order:
        order_fixed[elem] = count
        count += 1
    return order_fixed


def save_opt_band(n, result_file):
    """
    Save optimal bands for n graphs in GRAPH_DATA_PATH to result_file
    """
    path = RESULT_DATA_PATH + result_file
    with open(path, 'w') as f:
        header = ' '*150 + '\n'
        f.write(header)
        records = ''
        for i in range(n):
            file_path = '{}{}.txt'.format(GRAPH_DATA_PATH, i)
            cp = subprocess.Popen(['./opt_band/bin/MBandwidth',file_path,'3600','1'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
            lines = cp.stdout.readlines()
            opt_sequence = get_result(lines, NUM_NODES)
            opt_sequence = ';'.join(opt_sequence)
            line = '{};{}\n'.format(i,opt_sequence)
            records += line
            if i%BUFFER == 0:
                f.write(records)
                records = ''
        # last records
        if records:
            f.write(records)


def delete_graphs():
    shutil.rmtree(GRAPH_DATA_PATH[:-1])
    os.mkdir(GRAPH_DATA_PATH[:-1])


def load_opt_seq(file_name):
    path = RESULT_DATA_PATH + file_name
    with open(path,'r') as f:
        lines = f.readlines()
    d = dict()
    for line in lines[1:]: # removing header line
        v = [int(val) for val in line[:-1].split(';')] # remove final \n and parse to integers
        d[v[0]] = v[1:] # v[0] == idGraph, v[1:] == optimal sequence
    return d

    
def test_result(G, opt_seq, idx_file):
    """
    Comparison between heuristic and exact methods
    larger  = List with exact and heuristc results where exact went larger than heuristic
              structure: [idx,exact_band,heuristic_band]
    same    = Num of exact results that went same than heuristic
    smaller = Num of exact results that went smaller than heuristic
    """
    larger = list()
    same = 0
    smaller = 0

    ########################################################################
    # NOTE: Graph[0] in idx_file==0 has no edges, so it is not considered
    start = 1 if idx_file==0 else 0
    
    # a main roda um for loop, a cada iteração pega um bloco de grafos e roda os testes
    # para todos os grafos daquele bloco, como no primeiro bloco, o indice 0 do primeiro bloco não tem arestas
    # ele ignora. A partir dos outros blocos de grafos idx1 em diante (1.txt, 2.txt..), 
    # não tem mais esse problema e começa do 0
    for i in range(start,len(G)):
        rcm = list(reverse_cuthill_mckee_ordering(G[i]))
        heuristic_band = get_bandwidth('RCM', G[i], nodelist=rcm)
        cp_band = get_bandwidth('CP', G[i], nodelist=opt_seq[i])

        if cp_band < heuristic_band:
            smaller += 1
        elif cp_band == heuristic_band:
            same += 1
        else:
            larger.append([i,cp_band,heuristic_band])
    
    return larger,same,smaller


def save_larger_results(larger_result_file, larger):
    s = ''
    for l in larger:
        s += ';'.join(str(elm) for elm in l) + '\n'
        
    path = RESULT_DATA_PATH + '/larger_results/' + larger_result_file
    with open(path, 'w') as fr:
        fr.write(s)    


# Para cada arquivo n10_x.g6
for idx_file in range(initial_index_file, G6_DATA_NUM_BLOCKS):
    
    # Primeiro inicio de computo do tempo
    first_start = time.time()
    
    # Carrego os grafos do bloco
    src_file = 'n10_{}.g6'.format(idx_file)
    G = load_graphs(src_file)
    if not len(G[0].nodes) == NUM_NODES:
        raise 'Exception: graph nodes lenght different of parameter. Readed: {}, Expected: {}'.format(len(G[0].nodes), NUM_NODES)

    # Gravo os arquivos txt na pasta (um arquivo por grafo)
    for i in range(len(G)):
        write_txt_file(G[i],i)

    # Obtenho a banda com o programa em C e salvo a opt_sequence em arquivo (todas as opt_sequence de um bloco de grafos em um único arquivo)
    result_file = 'opt_seq_{}.txt'.format(src_file)
    save_opt_band(len(G), result_file)

    # Apago os grafos em txt
    delete_graphs()

    # Encerra o primeiro cômputo de tempo
    first_time = time.time() - first_start

    # Segundo inicio de cômputo do tempo
    second_start = time.time()

    # Carregando resultados
    opt_seq = load_opt_seq(result_file)

    # testando
    larger,same,smaller = test_result(G, opt_seq, idx_file)

    # Encerra o segundo cômputo de tempo
    second_time = time.time() - second_start

    # Escreve dados de cabeçalho no início do result_file:
    arr = [first_time,second_time,len(larger),same,smaller]
    s = ';'.join(str(elm) for elm in arr)
    path = RESULT_DATA_PATH + result_file
    with open(path, 'r+') as f:
        f.seek(0)
        f.write(s)

    # Se existem resultados exatos maiores que heurísticos, grava em disco
    if len(larger):
        larger_result_file = 'larger_{}.txt'.format(src_file)
        save_larger_results(larger_result_file, larger)

