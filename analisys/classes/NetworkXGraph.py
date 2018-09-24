import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from classes.Timer import Timer


class Graph:
    """A custom implementation of a NetworkX graph
    """
    
    def __init__(self):
        # basic attributes
        self._lines = []
        self._graph = None

        # dict of means of graph
        self._means = dict()
        
        # graphs used for analisys
        self._sample_graph, self._sub_graph = nx.Graph(), nx.Graph()
        
        # dict of measures get from analisys
        self._measures = dict()
        
        
        
    def build_from_tsv(self, tsv_file_path):
        """Build a graph
        
           Builds the graph and stores it in self._graph
           
           Args:
               tsv_file_path: the path of TSV file
        """
        Timer.start()

        self._read_lines_tsv(tsv_file_path)
        self._graph = nx.parse_edgelist(self._lines, nodetype = int, data=(('ajd_value',float),))
        self._means['num_triangles'] = int(sum(list(nx.triangles(self._graph).values()))/3)
        self._means['clustering_coefficient'] = nx.average_clustering(self._graph)
        self._means['diameter'] = nx.diameter(self._graph)
        
        print(nx.info(self._graph))
        print(('Number of lines: %s') % len(self._lines))
        print('Number of triangles: ', self._means['num_triangles'])
        print('Clustering coefficient: ', self._means['clustering_coefficient'])
        print('Diameter: ', self._means['diameter'])
        Timer.finish()
        
        
        
    def _read_lines_tsv(self, path, encoding='utf-8'):
        """Read a TSV file
        
           Loads the TSV file an stores it in self._lines
           
           Args:
               path: the path of TSV file
               encoding: default('utf-8')
        """
        with open(path, 'rb') as f:
            for line in f:
                self._lines.append(line.decode(encoding))
        f.closed
        
        
        
    def get_analisys_from_samples(self, measure_list):
        
        step = int(len(self._lines)/25)
        if step < 1:
            step = 1

        # Building through edge and node strategies
        for i in range(step, len(self._lines)+1, step):
            Timer.start()

            # getting edges from list with random choice
            sample_edge_list = np.random.choice(self._lines, i, replace=False)
            # creating a sample graph from sample_edge_list
            self._sample_graph = nx.parse_edgelist(sample_edge_list, nodetype = int, data=(('ajd_value',float),))
            # creating a subgraph of self._graph using nodes of the self._sample_graph
            self._sub_graph = self._graph.subgraph(self._sample_graph.nodes).copy()
            
            for measure in measure_list:
                self._calculate_measure(measure)

            elapsed = Timer.get_elapsed()
            print('Elapsed time for %d lines : %f, total nodes processed: %d' % 
                     (i, elapsed, self._sample_graph.number_of_nodes()))
            
        self._sample_graph.clear()
        self._sub_graph.clear()
        print()
        print('Analisys from samples finished. Call plot_analsys() to view results')
        
        
        
    def _calculate_measure(self, measure):
        if measure not in self._measures:
            self._measures[measure] = {'title': measure}
            self._measures[measure]['m'] = []
            self._measures[measure]['edge_list'] = []
            self._measures[measure]['node_list'] = []
                
        self._measures[measure]['m'].append(self._sample_graph.number_of_nodes())
        
        if measure == 'clustering_coefficient':
            self._measures[measure]['edge_list'].append(nx.average_clustering(self._sample_graph))
            self._measures[measure]['node_list'].append(nx.average_clustering(self._sub_graph))
        elif measure == 'diameter':
            self._measures[measure]['edge_list'].append(nx.diameter(self._sample_graph))
            self._measures[measure]['node_list'].append(nx.diameter(self._sub_graph))

            
            
    def plot_analisys(self, measure_list):
        
        for measure in measure_list:
        
            plt.plot(self._measures[measure]['m'], [self._means[measure]]*len(self._measures[measure]['m']), label='Total Graph')
            plt.plot(self._measures[measure]['m'], self._measures[measure]['edge_list'], label='edge list')
            plt.plot(self._measures[measure]['m'], self._measures[measure]['node_list'], label='node list')
            plt.legend()
            str_title = 'Comparison of %s between strategies' % self._measures[measure]['title']
            plt.title(str_title)
            plt.xlabel('Number of nodes')
            plt.ylabel(self._measures[measure]['title'])
            plt.show()
        

        