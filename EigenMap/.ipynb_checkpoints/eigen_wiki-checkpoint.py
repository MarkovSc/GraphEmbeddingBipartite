import numpy as np

from sklearn.linear_model import LogisticRegression

import networkx as nx
from networkx.algorithms import bipartite
from EigenMap import EigenMap

def read_edge_weight(file, sep='\t'):
    graph_data=dict()
    nodes_first = set()
    nodes_second = set()
    weight_edges = list()
    edgeDict=dict()
    node2index = dict()
    with open(file, encoding='utf-8') as file:
        for line in file:
            l = line.split(sep)[0:3]
            nodes_first.add(l[0])
            nodes_second.add(l[1])
            weight_edges.append(l)
    
    nodes = list(nodes_first) + list(nodes_second)
    node2index = dict([(node, index) for index, node in enumerate(nodes)])

    G = nx.Graph()
    G.add_nodes_from([index for index in range(len(nodes_first))], bipartite=0)
    G.add_nodes_from([index + len(nodes_first) for index in range(len(nodes_second))], bipartite=1)

    for l in weight_edges:
        G.add_edge(node2index[l[0]], node2index[l[1]], weight = int(l[2]))
    return nodes_first, nodes_second, node2index, G

if __name__ == "__main__":
    node_first, node_second, node2index, G = read_edge_weight("../Data/wiki/rating_train.dat")
    node_first_index = [node2index[node] for node in list(node_first)]
    node_second_index = [node2index[node] for node in list(node_second)]

    embed_u, embed_v = EigenMap(G).build()

    vector_first =[]
    for node in list(node_first):
        vector_first.append(node + ' ' + ' '.join(map(str, embed_u[node2index[node]])))

    vector_second =[]
    for node in list(node_second):
        vector_second.append(node + ' ' + ' '.join(map(str, embed_u[node2index[node]])))

    with open("../Data/wiki/EigenMap/vectors_u.dat", 'w') as file:
        for l in vector_first:
            file.write(l + '\n')

    with open("../Data/wiki/EigenMap/vectors_v.dat", 'w') as file:
        for l in vector_second:
            file.write(l + '\n')
