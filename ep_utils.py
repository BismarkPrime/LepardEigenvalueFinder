from time import time
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy import sparse
from typing import Dict, List, Tuple, Sequence

from utils import getSymmetricDifference

EPSILON = 1e-4

def _getEigenvaluesSparse(csc: sparse.csc_array, csr: sparse.csr_array, pi: Dict[int, List[int]], leps: List[List[int]], verbose=False) -> Tuple[List[float | complex], List[float | complex], Tuple[float, float, float]]:
    start_time = time()
    divisor_matrix = getDivisorMatrixSparse(csc, pi)
    div_time = time() - start_time
    if verbose:
        print(f"Divisor matrix computed in {div_time} seconds")

    # in practice, np.linalg.eigvals, scipy.linalg.eigvals, and scipy.linalg.eigvals(..., overwrite_a=True) run
    #   in roughly the same amount of time
    start_time = time()
    globals = getGlobals(divisor_matrix)
    globals_time = time() - start_time
    if verbose:
        print(f"Globals computed in {globals_time} seconds")

    start_time = time()
    locals = getLocals(csr, divisor_matrix, pi, leps)
    locals_time = time() - start_time
    if verbose:
        print(f"Locals computed in {locals_time} seconds")

    return globals, locals, (div_time, globals_time, locals_time)


def getGlobals(divisor_matrix: np.ndarray) -> List[float | complex]:
    return np.linalg.eigvals(divisor_matrix).tolist()


def getLocals(csr: sparse.csr_array, divisor_matrix: np.ndarray, pi: Dict[int, List[int]], leps: List[List[int]]) -> List[float | complex]:
    
    # For each LEP:
    #    a. Create subgraph
    #    b. Compute divisor graph of subgraph
    #    c. Calculate spectrum of subgraph, divisor graph
    #    d. Compute difference eigs(SG) - eigs(DG)
    locals = []
    for lep in leps:
        nodes: List[int] = []
        for V in lep:
            nodes.extend(pi[V])
        # skip iterations for which globals = locals
        if len(nodes) < 2:
            continue

        subgraph = csr[nodes,:][:,nodes]
        divisor_submatrix = divisor_matrix[lep,:][:,lep]

        subgraph_globals = np.linalg.eigvals(divisor_submatrix)
        subgraph_locals = np.linalg.eigvals(subgraph.todense())

        locals.extend(getSetDifference(list(subgraph_locals), list(subgraph_globals)))
    return locals
    

def getDivisorMatrixSparse(mat_csc: sparse.csc_array, pi: Dict[int, List[int]]) -> np.ndarray:

    node2ep = { node: i for i, V in pi.items() for node in V }
    div_mat = np.zeros((len(pi), len(pi)), dtype=int)

    for i, V in pi.items():
        node = V[0]
        neighbors = mat_csc.indices[mat_csc.indptr[node]:mat_csc.indptr[node + 1]]
        for neighbor in neighbors:
            div_mat[i, node2ep[neighbor]] += 1 # perhaps += weight for weighted graphs...
    
    return div_mat


def getSetDifference(list1: Sequence[float | complex], list2: Sequence[float | complex], epsilon_start=EPSILON, epsilon_max=1e-1) -> List[complex]:
    return getSymmetricDifference(list1, list2, epsilon_start=epsilon_start, epsilon_max=epsilon_max)[0]
    

# GENERAL HELPER FUNCTIONS

def plotEquitablePartition(G, pi, pos_dict=None):
    """Plots the equitable partition of a graph, with each element in its own color.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to be plotted
        pi : dict
            The equitable partition of the graph, as returned by ep_finder
        pos_dict : dict (optional)
            A dictionary mapping nodes to their x,y coordinates. Only used when a such
            values are available and meaningful (such as a random geometric graph).
    """
    # iterator over equidistant colors on the color spectrum
    color = iter(plt.cm.get_cmap("rainbow")(np.linspace(0, 1, len(pi) + 1))) # type: ignore
    # stores the color for each node
    default_color = next(color)
    color_list = [default_color for _ in range(G.number_of_nodes())]
    # assign all vertices in the same partition element to the same color
    for V_i in pi.values():
        c = next(color)
        for vertex in V_i:
            color_list[vertex] = c
    
    if pos_dict is None:
        # layout options include: spring, random, circular, spiral, spring, kamada_kawai, etc
        pos_dict = nx.kamada_kawai_layout(G)

    plt.ion()

    nx.draw_networkx(G, pos=pos_dict, node_color=color_list)
    plt.show()
    noop = 1
    # need to pause briefly because GUI events (e.g., drawing) happen when the main loop sleeps
    plt.pause(.001)
