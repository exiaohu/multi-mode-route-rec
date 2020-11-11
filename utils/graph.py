import numpy as np
import networkx as nx
import pymetis

from .helper import timing


@timing
def graph_partition(g: nx.DiGraph, n_part: int):
    edges = ((node, np.array(list(edges))) for node, edges in g.to_undirected().adjacency())
    nodes, adj_list = zip(*sorted(edges, key=lambda i: i[0]))

    _, membership = pymetis.part_graph(n_part, adj_list)

    partitions = [set() for _ in range(n_part)]
    for node, member in zip(nodes, membership):
        partitions[member].add(node)

    return list(map(sorted, partitions)), {node: member for node, member in zip(nodes, membership)}
