import numpy as np
import networkx as nx
from scipy import sparse

from typing import Any, List, Set, Dict

# POTENTIAL OPTIMIZATIONS:
#   Using Disjoint Set data structures to store partitions

def initFromNx(G: nx.Graph | nx.DiGraph) -> List[Set[Any]]:
    """Initializes the inverted neighbor dictionary required to compute leps.
    
    ARGUMENTS:
        G : The graph to analyzed
    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """

    # NOTE: N stores the in-edge neighbors, i.e. N[v] contains all nodes w with an edge w -> v.
    #    Thus, it is different than just calling G.neighbors(v) for directed graphs.
    N = [set(G.predecessors(node) if type(G) is nx.DiGraph else G.neighbors(node)) for node in G.nodes()]

    return N

def initFromSparse(mat: sparse.csc_array) -> List[Set[Any]]:
    """Initializes the inverted neighbor dictionary required to compute leps.
    
    ARGUMENTS:
        G : The graph to analyzed
    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """

    if type(mat) is not sparse.csc_array:
        raise ValueError("Input matrix must be in CSC format.")
    N = [set(mat.indices[mat.indptr[i]:mat.indptr[i + 1]]) for i in range(mat.shape[0])]
    
    return N

def getLocalEquitablePartitions(N: List[Set[Any]], pi: Dict[int, List[Any]]) -> List[List[int]]:
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        N :     A dictionary containing nodes as keys with their in-edge neighbors as values
        pi :    The equitable partition of the graph, as returned by ep_finder
    
    RETURNS:
        A list of sets, with each set containing the indices/keys of partition elements
            that can be grouped together in the same local equitable partition
    """
    # dict that maps nodes to their partition element
    partition_dict = np.empty(len(N), dtype=int)
    for i, V in pi.items():
        for node in V:
            partition_dict[node] = i

    # keeps track of which partition elements are stuck together by internal cohesion,
    #   with partition element index as key and internally cohesive elements as values
    lep_network = dict()

    for i, V in pi.items():
        common_neighbors = set(N[V[0]])
        for v in V:
            common_neighbors.intersection_update(N[v])
        for v in V:
            for unique_neighbor in set(N[v]) - common_neighbors:
                __link(i, partition_dict[unique_neighbor], lep_network) # type: ignore

    leps = __extractConnectedComponents(lep_network, len(pi))
    # convert to List of Lists to be consistent with EPFinder
    lep_list = [[int(l) for l in lep] for lep in leps]
    return lep_list

def __link(i: int, j: int, edge_dict: Dict[int, Set[int]]) -> None:
    if i not in edge_dict:
        edge_dict.update({i: set()})
    edge_dict.get(i).add(j) # type: ignore

    if j not in edge_dict:
        edge_dict.update({j: set()})
    edge_dict.get(j).add(i) # type: ignore

def __extractConnectedComponents(edge_dict: Dict[int, Set[int]], num_nodes: int) -> List[Set[int]]:
    visited = set()
    scc_list = []
    for i in range(num_nodes):
        if i not in visited:
            scc = set()
            scc.add(i)
            visited.add(i)
            neighbors = edge_dict.get(i)
            while neighbors is not None and len(neighbors) > 0:
                j = neighbors.pop()
                scc.add(j)
                visited.add(j)
                neighbors.update((edge_dict.get(j) or set()).difference(scc))
            scc_list.append(scc)
    return scc_list