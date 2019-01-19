import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from networkx.utils import reverse_cuthill_mckee_ordering
from classes.Timer import Timer


class Graph:
    """A custom implementation of a NetworkX graph
    """
    
    def __init__(self):
        # basic attributes
        self._lines = list()
        self._graph = None
        self._is_connected = False

        # dict of means of self._graph
        self._means = dict()
        
        # sample graph used for analisys
        self._sample_graph = nx.Graph()
        
        # dict of measures get from analisys
        self._measures = dict()
        
        # list of subgraphs of self._graph
        self._subgraphs = list()
        
        
        
    def build_from_tsv(self, tsv_file_path, data=None, show_info=None):
        """Build a graph
        
           Builds the graph and stores it in self._graph
           
           Args:
               tsv_file_path: the path of TSV file
        """
        Timer.start()

        self._read_lines_tsv(tsv_file_path)
        #self._graph = nx.parse_edgelist(self._lines, nodetype = int, data=(('ajd_value',float),))
        self._graph = nx.parse_edgelist(self._lines, nodetype = int, data=data)
        self._is_connected = nx.is_connected(self._graph)

        print(('Number of lines read from TSV file: %s') % len(self._lines))
        print(nx.info(self._graph))
        print('Connected: ', '%s' % 'Yes' if self._is_connected else 'No')
        
        if show_info is not None:
            if 'num_triangles' in show_info:
                self._means['num_triangles'] = int(sum(list(nx.triangles(self._graph).values()))/3)
                print('Number of triangles: ', self._means['num_triangles'])
                
            if 'clustering_coefficient' in show_info:
                self._means['clustering_coefficient'] = nx.average_clustering(self._graph)
                print('Clustering coefficient: ', self._means['clustering_coefficient'])
                
            if 'bandwidth' in show_info:
                self._means['bandwidth'] = self._get_bandwidth(self._graph)
                self._means['reduced_bandwidth'] = self._get_reduced_bandwidth(self._graph)
                print('Bandwidth: ', self._means['bandwidth'])
                print('Reduced Bandwidth: ', self._means['reduced_bandwidth'])

            if 'diameter' in show_info:
                if(self._is_connected):
                    self._means['diameter'] = nx.diameter(self._graph)
                else:
                    self._means['diameter'] = self._get_diameter_from_disconnected(self._graph)
                print('Diameter: ', self._means['diameter'])
                
            if 'radius' in show_info:
                if(self._is_connected):
                    self._means['radius'] = nx.radius(self._graph)
                else:
                    self._means['radius'] = self._get_radius_from_disconnected(self._graph)
                print('Radius: ', self._means['radius'])
        
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
        
        print('----------------------------------------')
        print('Analysing by random edge sample strategy')
        print('----------------------------------------')
        
        # Calculating length of step
        step = int(len(self._lines)/25)
        if step < 1:
            step = 1

        # Building and analysing
        for i in range(step, len(self._lines)+1, step):
            Timer.start()
            # getting edges from list with random choice
            sample_edge_list = np.random.choice(self._lines, i, replace=False)
            # creating a sample graph from sample_edge_list
            self._sample_graph = nx.parse_edgelist(sample_edge_list, nodetype = int, data=(('ajd_value',float),))
            # calculating measures
            for measure in measure_list:
                self._calculate_measure(measure, 'edge_list', self._sample_graph)
            # finishing
            elapsed = Timer.get_elapsed()
            print('Elapsed time for %d edges: %f, total nodes processed: %d' % 
                     (i, elapsed, self._sample_graph.number_of_nodes()))
            
        # emptying memory
        sample_edge_list = None
        self._sample_graph.clear()
        
        
        print('----------------------------------------')
        print('Analysing by random node sample strategy')
        print('----------------------------------------')
            
        # Calculating length of step
        nn = self._graph.number_of_nodes()
        step = int(nn/25)
        if step < 1:
            step = 1

        # Building and analysing
        for i in range(step, nn+1, step):
            Timer.start()
            # getting nodes from self._graph with random choice
            sample_node_list = np.random.choice(self._graph.nodes, i, replace=False)
            # creating a sample graph from sample_node_list
            self._sample_graph = self._graph.subgraph(sample_node_list).copy()
            # calculating measures
            for measure in measure_list:
                self._calculate_measure(measure, 'node_list', self._sample_graph)
            # finishing
            elapsed = Timer.get_elapsed()
            print('Elapsed time for %d nodes: %f, total nodes processed: %d' % 
                     (i, elapsed, self._sample_graph.number_of_nodes()))
        
        # emptying memory
        sample_node_list = None
        self._sample_graph.clear()
        
        
        print('----------------------------------------')
        print('Analysing by random walk sample strategy')
        print('----------------------------------------')
        
        # initializing
        visited = set()
        
        # To control the point of plotting
        j = 0
        checkpoint = int(self._graph.number_of_nodes()/25)
        if checkpoint < 1:
            checkpoint = 1

        # Let's walk k steps
        k = self._graph.number_of_nodes() * 6

        # starting random
        idNode = np.random.choice(self._graph.nodes, 1)[0]
        visited.add(idNode)
        
        # Building and analysing
        for i in range(k):
            Timer.start()
            # getting nodes from self._graph with random walk
            neighbors = [n for n in self._graph.neighbors(idNode)]
            idNode = np.random.choice(neighbors, 1)[0]
            visited.add(idNode)
            j = j + 1
            if(j == checkpoint):
                j = 0
                # creating a sample graph from visited_list
                self._sample_graph = self._graph.subgraph(visited).copy()
                # calculating measures
                for measure in measure_list:
                    self._calculate_measure(measure, 'walk_list', self._sample_graph)
                # finishing
                elapsed = Timer.get_elapsed()
                print('Elapsed time for %d steps: %f, total nodes processed: %d' % 
                         (i+1, elapsed, self._sample_graph.number_of_nodes()))
        
        # emptying memory
        visited = None
        self._sample_graph.clear()
        
        
        # finishing the analysis
        print()
        print('Analisys from samples finished. Call plot_analsys() to view results')
        
        
        
    def _calculate_measure(self, measure, strategy, graph):
        if measure not in self._measures:
            self._measures[measure] = {'title': measure}
            self._measures[measure]['edge_list'] = {'m': [], 'data': []}
            self._measures[measure]['node_list'] = {'m': [], 'data': []}
            self._measures[measure]['walk_list'] = {'m': [], 'data': []}
                
        self._measures[measure][strategy]['m'].append(graph.number_of_nodes())
        
        if measure == 'clustering_coefficient':
            self._measures[measure][strategy]['data'].append(nx.average_clustering(graph))
        
        elif measure == 'diameter':
            if nx.is_connected(graph):
                self._measures[measure][strategy]['data'].append(nx.diameter(graph))
            else:
                self._measures[measure][strategy]['data'].append(self._get_diameter_from_disconnected(graph))
                
        elif measure == 'radius':
            if nx.is_connected(graph):
                self._measures[measure][strategy]['data'].append(nx.radius(graph))
            else:
                self._measures[measure][strategy]['data'].append(self._get_radius_from_disconnected(graph))
                
        elif measure == 'reduced_bandwidth':
            self._measures[measure][strategy]['data'].append(self._get_reduced_bandwidth(graph))

            
    def plot_analisys(self, measure_list):
        
        for measure in measure_list:
        
            # Config of axis
            # v = [xmin, xmax, ymin, ymax]
            # v = [0, 11, 0, 5]
            # plt.axis(v)
            
            plt.plot(np.arange(1, self._graph.number_of_nodes() + 1), [self._means[measure]]*self._graph.number_of_nodes(), label='Total Graph')
            plt.plot(self._measures[measure]['edge_list']['m'], self._measures[measure]['edge_list']['data'], label='edge list')
            plt.plot(self._measures[measure]['node_list']['m'], self._measures[measure]['node_list']['data'], label='node list')
            plt.plot(self._measures[measure]['walk_list']['m'], self._measures[measure]['walk_list']['data'], label='random walk')
            plt.legend()
            str_title = 'Comparison of %s between strategies' % self._measures[measure]['title']
            plt.title(str_title)
            plt.xlabel('Number of nodes')
            plt.ylabel(self._measures[measure]['title'])
            plt.show()

            
        
    def _get_diameter_from_disconnected(self, g):
        max_diameter = 0

        for c in nx.connected_components(g):
            sg = g.subgraph(c).copy()
            max_diameter = max(max_diameter, nx.diameter(sg))
            
        sg.clear()
        return max_diameter
            
        
        
    def _get_radius_from_disconnected(self, g):
        max_radius = 0

        for c in nx.connected_components(g):
            sg = g.subgraph(c).copy()
            max_radius = max(max_radius, nx.radius(sg))

        sg.clear()
        return max_radius
    
    
    
    def get_degree(self):
        degree = dict()
        degree.update(nx.degree(self._graph))
        return degree
    
    
    
    def transform_subgraph_list(self, delete_original = False):
        if self._is_connected:
            raise Exception('Transformation not executed. Graph is connected.')
            
        for c in nx.connected_components(self._graph):
            self._subgraphs.append(self._graph.subgraph(c).copy())
            
        if delete_original:
            self._graph.clear()
            print('Original graph deleted.')
            
        print('%d subgraphs created.' % len(self._subgraphs))
    
    
    
    def get_eccentricity(self):
        
        ecc = dict()
                    
        if self._is_connected:
            ecc.update(nx.eccentricity(self._graph))
        
        else:
            if not self._subgraphs:
                for c in nx.connected_components(self._graph):
                    sg = self._graph.subgraph(c).copy()
                    ecc.update(nx.eccentricity(sg))
            else:
                for sg in self._subgraphs:
                    ecc.update(nx.eccentricity(sg))
        
        return ecc
    
    
    
    def get_position(self):
        
        pos = dict()
        
        if self._is_connected:
            pos.update(self._get_list_pos(self._graph))
        
        else:
            if not self._subgraphs:
                for c in nx.connected_components(self._graph):
                    sg = self._graph.subgraph(c).copy()
                    pos.update(self._get_list_pos(sg))
            else:
                for sg in self._subgraphs:
                    pos.update(self._get_list_pos(sg))
                
        return pos
    
    
    
    def _get_list_pos(self, graph):
        
        temp_pos = dict()
        
        center = nx.center(graph)
        for node in center:
            temp_pos[node] = 1

        periphery = nx.periphery(graph)
        for node in periphery:
            temp_pos[node] = 2
            
        return temp_pos
    
    
    
    def _get_bandwidth(self, graph, rcm=None):
        A = nx.laplacian_matrix(graph, nodelist=rcm)
        x, y = np.nonzero(A)
        return (y - x).max() + (x - y).max() + 1
    
    
    
    def _get_reduced_bandwidth(self, graph):
        if len(graph.edges()) == 0:
            return 0
        rcm = list(reverse_cuthill_mckee_ordering(graph))
        return self._get_bandwidth(graph, rcm)

    