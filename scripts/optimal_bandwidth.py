import time
import sys
#from sys import getsizeof
import networkx as nx

if len(sys.argv) != 2:
	print("USAGE: {} [graph6_path]".format(sys.argv[0]))
	quit()

start = time.time()
G = nx.read_graph6(sys.argv[1])
end = time.time()
print('Time: {}'.format(end-start))
print('Length G: {}'.format(len(G)))
