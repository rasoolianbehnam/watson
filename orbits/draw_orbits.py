import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

file_name_tmp = 'orbit%02d.txt'
graphlets = []
for count in range(73):
    file_name = file_name_tmp%count
    f = open(file_name, 'r')
    n = 0
    for line in f:
        splitted = line.strip().replace(" ", "").split(",")
        if len(splitted) != 2:
            continue
        m = np.max([int(i) for i in splitted])
        n = np.max([m, n])
    f.close()
    G = nx.Graph()
    G.add_nodes_from([str(i) for i in range(n)])
    f = open(file_name, 'r')
    for line in f:
        splitted = line.strip().replace(" ", "").split(",")
        if len(splitted) != 2:
            continue
        G.add_edge(splitted[0], splitted[1])
    graphlets.append(G)
#plt.show()
fig, ax = plt.subplots(3, 30, num=1)
for i, G in enumerate(graphlets):
    nx.draw(G, ax=ax[i//30, i%30],node_size=range(n), pos = nx.spring_layout(G, scale=3))
plt.show()
