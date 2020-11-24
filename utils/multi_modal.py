import networkx as nx

from .gen_nets import *
from .routing import EdgesLookup

__all__ = ['get_edges_by_modals']

cache = dict()


def for_modals(modals):
    global cache

    def wrap(f):
        def wrapped_f(*args, **kwargs):
            print('BEGIN: constructing a traffic network of', modals)

            if tuple(sorted(modals)) not in cache:
                net = f(*args, **kwargs)
                cache[tuple(sorted(modals))] = net
            else:
                net = cache[tuple(sorted(modals))]

            print('END: constructing a traffic network of', modals)
            return net

        return wrapped_f

    return wrap


def get_edges_by_modals(modals) -> EdgesLookup:
    global cache

    assert tuple(sorted(modals)) in cache.keys(), f'Modal combination {modals} is not supported yet.'

    return cache.get(tuple(sorted(modals)))


@for_modals(['walking'])
def walking() -> EdgesLookup:
    net = get_road()
    net = net.subgraph(filter(lambda n: n.modal == 'walk', net.nodes))
    largest_cc = max(nx.weakly_connected_components(net), key=len)
    net = net.subgraph(largest_cc)
    return EdgesLookup([(o, d, net.edges[(o, d)]) for o, d in net.edges], None)


@for_modals(['walking', 'driving'])
def driving() -> EdgesLookup:
    net = get_road()
    return EdgesLookup([(o, d, net.edges[(o, d)]) for o, d in net.edges], None)


@for_modals(['walking', 'taxi'])
def taxi() -> EdgesLookup:
    net = get_road()
    return EdgesLookup([(o, d, net.edges[(o, d)]) for o, d in net.edges], None)


@for_modals(['walking', 'public'])
def public() -> EdgesLookup:
    net = get_multi_modal()
    net = net.subgraph(filter(lambda n: n.modal in ('walk', 'bus', 'subway'), net.nodes))
    return EdgesLookup([(o, d, net.edges[(o, d)]) for o, d in net.edges], None)


walking()
# driving()
# taxi()
# public()
