import sys
sys.path.append('..')
from classes.NetworkXGraph import Graph

# user parameters
import argparse

class Param():
    pass

PARAM = Param()
parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='STR dataset file name')
parser.parse_args(namespace=PARAM)

info = ['clustering_coefficient', 'diameter', 'radius']
file_path = '../datasets/{}'.format(PARAM.file)

g = Graph()

g.build_from_tsv(file_path, data=(('ajd_value',float),), show_info=info)

g.get_analisys_from_samples(['clustering_coefficient', 'diameter', 'radius'])

g.plot_analisys(['clustering_coefficient', 'diameter', 'radius'], PARAM.file)