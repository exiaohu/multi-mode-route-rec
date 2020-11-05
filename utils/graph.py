import networkx as nx
import numpy as np
import pymetis

from .helper import timing


@timing
def graph_partition(g: nx.DiGraph, n_part: int):
    nodes, adj_list = zip(*[(node, list(edges)) for node, edges in g.adjacency()])

    adj_list = list(map(lambda edges: np.array(list(map(lambda edge: nodes.index(edge), edges))), adj_list))

    _, membership = pymetis.part_graph(n_part, adj_list)

    partitions = [set() for _ in range(n_part)]
    for node, member in zip(nodes, membership):
        partitions[member].add(node)

    return partitions
