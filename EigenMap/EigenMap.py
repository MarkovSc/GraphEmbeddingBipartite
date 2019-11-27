# coding: utf-8
"""
Author:
    markov yongsaima@163.com

"""

import networkx as nx
import numpy as np
import scipy.sparse.linalg as slin

class EigenMap():
    def __init__(self, graph, emb_dim = 128):
        self.graph = graph
        self.idx2node = list(self.graph.nodes())
        self.node2idx = dict([(self.idx2node[index], index) for index in range(len(self.idx2node))])
        self.node_size = self.graph.number_of_nodes()
        self.emb_dim = emb_dim
    
    # simple m * n  or (m + n) * (m + n)
    def build(self):
        A = nx.to_scipy_sparse_matrix(self.graph)
        rows, cols = A.nonzero()
        A_ = A.copy()
        A_[cols, rows] = A[rows, cols]
        A_ = A_.toarray()
        D = np.zeros_like(A_)
        for i in range(self.node_size):
            D[i][i] = np.sum(A_[i])
        L = D - A_
        u, s, vt = slin.svds(L.astype(float), k=self.emb_dim, which='LM')
        return u, vt
    def build_adj(self):
        A = nx.to_scipy_sparse_matrix(self.graph)
        rows, cols = A.nonzero()
        A_ = A.copy()
        A_[cols, rows] = A[rows, cols]
        A_ = A_.toarray()
        u, s, vt = slin.svds(A_.astype(float), k=self.emb_dim, which='LM')
        return u, vt
