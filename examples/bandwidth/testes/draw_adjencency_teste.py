# from matplotlib import pyplot as plt

data = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]



# plt.imshow(data, cmap='gray', interpolation='none')
# plt.show()

# fig = plt.figure(figsize=(5, 5)) # in inches
# plt.imshow(data, cmap="gray", interpolation="none")
# plt.show()


import networkx as nx
from matplotlib import pyplot, patches
import numpy as np
from PIL import Image

G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3)])
adj = nx.to_numpy_matrix(G)

w, h = 3, 3
data = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(len(adj)):
    for j in range(len(adj)):
        if adj[i, j] == 1:
            # data[i:170, j:170] = np.array([255, 255, 255])
            data[i, j] = np.array([255, 255, 255])
# data[0:256, 0:256] = [255, 255, 255] # red patch in upper left
img = Image.fromarray(data, 'RGB')
fixed_height = 1024
height_percent = (fixed_height / float(img.size[1]))
width_size = int((float(img.size[0]) * float(height_percent)))
resized = img.resize((512, 512), Image.NEAREST)
print(resized.size)
# resized.save('my.png')
resized.show()

pyplot.imshow(adj, cmap="gray")
pyplot.xticks([])
pyplot.yticks([])
pyplot.show()
# pyplot.legend("oiiiiiiiiii")
# pyplot.savefig('adj.png', legend=None)

# img2 = Image.fromarray(adj, (3, 3), 'RGB')
# img2.save('img2.png')