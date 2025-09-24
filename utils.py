import os
import networkx as nx
from scipy import sparse as sp
import cmath

from typing import List, Tuple, Sequence

# for debugging purposes
from matplotlib import pyplot as plt


def oneGraphToRuleThemAll(file_name: str, directed: bool=False, cust_com: str="#", weighted: bool=False) -> sp.csr_array:
    """
    Detects the type of input graph. Reads it in and outputs it as a sparse matrix in CSR format.
    PARAMETERS
    ---------------------------------------------
        file_name (str): name of the file to load into the graph
    """

    def start_section(content: str) -> None:
        border = '#' * os.get_terminal_size().columns
        pad_left = (os.get_terminal_size().columns - len(content)) // 2
        print(f"{border}\n\n{' ' * pad_left}{content.upper()}\n\n{border}")
    
    extension = file_name.split('.')[-1]
    match extension.lower():
        case 'csv': 
            start_section("CSV File Detected")
            print("ASSUMPTIONS:\nThis txt file contains edge data of the form\n" + \
                                "\tsrc_label,dst_label\n" + \
                                "\tsrc_label,dst_label\n" + \
                                "\t...\n")
            G = nx.read_edgelist(file_name, delimiter=',', create_using=nx.DiGraph if directed else nx.Graph)
            G_sparse = nx.to_scipy_sparse_array(G, format='csr', dtype='b')

        case 'txt':
            start_section("TXT File Detected")
            print("ASSUMPTIONS:\nThis txt file contains edge data of the form\n" + \
                                "\tsrc_label dst_label\n" + \
                                "\tsrc_label dst_label\n" + \
                                "\t...\n")
            G = nx.read_edgelist(file_name, create_using=nx.DiGraph if directed else nx.Graph)
            G_sparse = nx.to_scipy_sparse_array(G, format='csr', dtype='b')
            
        case 'graphml':
            start_section("GraphML File Detected")
            G = nx.read_graphml(file_name)
            G_sparse = nx.to_scipy_sparse_array(G, format='csr', dtype='b')
            
        case 'edgelist':
            start_section("EDGELIST File Detected")
            if weighted: 
                G = nx.read_edgelist(file_name, comments=cust_com, data=(('weight',float),), create_using=nx.DiGraph if directed else nx.Graph) # type: ignore
            else:
                G = nx.read_edgelist(file_name, comments=cust_com, create_using=nx.DiGraph if directed else nx.Graph)
            G_sparse = nx.to_scipy_sparse_array(G, format='csr')

        case 'edges':
            start_section("EDGES File Detected")
            if weighted: 
                G = nx.read_edgelist(file_name, comments=cust_com, data=(('weight',float),), create_using=nx.DiGraph if directed else nx.Graph) # type: ignore
            else:
                G = nx.read_edgelist(file_name, comments=cust_com, create_using=nx.DiGraph if directed else nx.Graph)
            G_sparse = nx.to_scipy_sparse_array(G, format='csr')
        
        case _:
            raise ValueError(f"Unsupported file extension: {extension}. Supported extensions are: csv, txt, graphml, edgelist, edges.")

    return G_sparse


def getSymmetricDifference(list1: Sequence[complex], list2: Sequence[complex], epsilon_start=1e-4, epsilon_max=1e-1) -> Tuple[List[complex], List[complex]]:
    '''
        Gets the symmetric difference of two lists. (Returns list1 - list2, list2 - list1)
        Assumes that two elements are equal if they are within epsilon of one another.
        Increases the value of epsilon by an order of magnitude, up to epsilon_max, until all elements of list2 can be removed from list1.
        If elements remain in list2 after reaching epsilon_max, throws an error.
    '''
    # NOTE: since maximum bipartite matching is generally solved as a max-flow problem, it
    #   may be comparable in speed to this naive method (n^2), and is more robust to edge cases.
    #   (Note, however, that the current implementation of max flow in NetworkX is quite slow.)
    #   Consider defaulting to the bipartite method in getSymmetricDifferenceMatching
    
    res1 = []
    res2 = []

    skip_indices1 = set()
    skip_indices2 = set()
    epsilon = epsilon_start

    while len(skip_indices2) < len(list2):
        for j, cnum2 in enumerate(list2):
            if j in skip_indices2:
                continue
            for i, cnum1 in enumerate(list1):
                if i in skip_indices1:
                    continue
                if cmath.isclose(cnum2, cnum1, abs_tol=epsilon):
                    skip_indices1.add(i)
                    skip_indices2.add(j)
                    break
        
        # if, with epsilon = epsilon_max, we still can't perform the operation, raise an error
        if epsilon >= epsilon_max and len(skip_indices2) < len(list2):
            unique_to_list1 = [cnum for i, cnum in enumerate(list1) if i not in skip_indices1]
            unique_to_list2 = [cnum for i, cnum in enumerate(list2) if i not in skip_indices2]
            print("Elements of list2 not present in list1:\n" +
                            f"{unique_to_list2}\n" +
                            "Elements of list1 not present in list2:\n" +
                            f"{unique_to_list1}" +
                            "Consider increasing epsilon_max.")
            
            # for debugging purposes
            # plot all points in list1 and list2
            plt.plot([cnum.real for cnum in list1], [cnum.imag for cnum in list1], 'ro')
            # don't fill in these points so we can see the overlap
            plt.plot([cnum.real for cnum in list2], [cnum.imag for cnum in list2], 'bo', fillstyle='none')
            plt.show()

            noop = 1 # breakpoint here
            # plot points in list1 that are not in list2 and vice versa
            plt.plot(*zip(*[(cnum.real, cnum.imag) for cnum in unique_to_list1]), 'ro')
            plt.plot(*zip(*[(cnum.real, cnum.imag) for cnum in unique_to_list2]), 'bo', fillstyle='none')
            plt.show()
            
            raise Exception("Elements of list2 not present in list1:\n" +
                            f"{[cnum for i, cnum in enumerate(list2) if i not in skip_indices2]}\n" +
                            "Elements of list1 not present in list2:\n" +
                            f"{[cnum for i, cnum in enumerate(list1) if i not in skip_indices1]}" +
                            "Consider increasing epsilon_max.")
        # double epsilon until we reach epsilon_max
        epsilon = min(epsilon * 2, epsilon_max)
        

    for i, cnum in enumerate(list1):
        if i not in skip_indices1:
            res1.append(cnum)
    
    for j, cnum in enumerate(list2):
        if j not in skip_indices2:
            res2.append(cnum)

    return res1, res2


def getSymmetricDifferenceMatching(list1: List[complex], list2: List[complex], abs_tol=5e-2) -> Tuple[List, List]:
    '''
        Gets the symmetric difference of two lists of complex numbers using a bipartite 
        matching algorithm to find the maximum number of complex pairs (c1, c2), where 
        c1 is from list1 and c2 from list2, such that c1 and c2 are sufficiently close to
        be considered equal complex numbers given some floating point tolerance.
        Returns two lists: list1 - list2, and list2 - list1
    '''
    # 1. create a bipartite graph with edges between each complex number in list1 and all 
    #   complex numbers in list2 that are sufficiently close to be considered equal
    #   (within some floating-point tolerance)
    sparse_graph = sp.dok_matrix((len(list1), len(list2)), dtype=bool)
    for i, cnum1 in enumerate(list1):
        for j, cnum2 in enumerate(list2):
            if cmath.isclose(cnum1, cnum2, abs_tol=abs_tol):
                sparse_graph[i, j] = True
    # 2. find the maximum matching of the graph
    matches = sp.csgraph.maximum_bipartite_matching(sparse_graph.tocsr(), perm_type='column')
    # 3. return the symmetric difference of the two lists, where the elements in the
    #   matching are removed from the lists
    skipIndices1 = set()
    skipIndices2 = set()
    for i, j in enumerate(matches):
        if j != -1:
            skipIndices1.add(i)
            skipIndices2.add(j)
    res1 = []
    res2 = []
    for i, cnum in enumerate(list1):
        if i not in skipIndices1:
            res1.append(cnum)
    for j, cnum in enumerate(list2):
        if j not in skipIndices2:
            res2.append(cnum)
    return res1, res2