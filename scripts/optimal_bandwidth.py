import time
import sys
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
import numpy as np

if len(sys.argv) != 3:
	print("USAGE: {} [graph6_path] [permuts_path]".format(sys.argv[0]))
	quit()

def get_bandwidth(G,rcm):
	L = nx.laplacian_matrix(G, nodelist=rcm) # Laplacian Matrix
	x,y = np.nonzero(L)
	return (y-x).max()+(x-y).max()+1

# result file name
result_file = sys.argv[1].split('/')[-1].split('.')[0] + '_result.txt'

# loading graphs
G = nx.read_graph6(sys.argv[1])

# loading permuts
with open(sys.argv[2]) as f:
    a = f.readlines()

for i in range(1,37):

	# initial rcm and bandwidth
	initial_rcm = list(reverse_cuthill_mckee_ordering(G[i]))
	initial_band = get_bandwidth(G[i], initial_rcm)
	print('Reduced bandwidth of G{}: {} with rcm {}'.format(i,initial_band, initial_rcm))

	# initializing controllers
	min_rcm = initial_rcm
	min_band = initial_band
	same_band = 0
	smaller_band = 0
	start = time.time()
	total_start = time.time()

	count = 0
	for r in a:
		rcm = [int(n) for n in r.split(';')]
		band = get_bandwidth(G[i], rcm)

		if band < min_band:
			print('-----------------------------------------------------------------------------------')
			print('min_bad {} replaced for {}'.format(min_band, band))
			min_band = band
			min_rcm = rcm
			if band < initial_band:
				smaller_band += 1

		if band == initial_band:
			same_band += 1

		count += 1
		if count % 300000 == 0:
			print('300.000 processed in {}'.format(time.time() - start))
			start = time.time()
			count = 0

	print('-----------------------------------------------------------------------------------')
	print('Reduced bandwidth of G{}: {} with rcm {}'.format(i,min_band,min_rcm))
	print('Same initial band found: {}'.format(same_band))
	print('Smaller band found: {}'.format(smaller_band))

	with open(result_file,'a') as f:
		f.write(i)
		f.write(G[i].nodes)
		f.write(G[i].edges)
		f.write('Initial bandwidth of G{}: {} with rcm {}'.format(i,initial_band,initial_rcm))
		f.write('Reduced bandwidth of G{}: {} with rcm '.format(i,min_band,min_rcm))
		f.write('Same initial band found: {}'.format(same_band))
		f.write('Smaller band found: {}'.format(smaller_band))

	print('Results recorded in {}'.format(result_file))
	print('Total time for G{}: {}'.format(i, time.time() - total_start))
