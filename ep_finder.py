# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)

To improve readability, we have changed some of the variable names used in Bastert's
paper. To translate between our naming and his, we provide the following table:

NAME IN PAPER       || NAME IN CODE
====================||====================
f (color)           || old_color
f bar (pseudo color)|| new_color
hit                 || curr_color_neighbors
p (structure_value) || out_edge_count
current_p           || curr_conns
current_color       || split_color

"""

from typing import Any, List, Set, Dict
import scipy.sparse as sp
import networkx as nx

class LinkedListNode:
    """Base class for doubly-linked list nodes"""
    __slots__ = 'next', 'prev'

    def __init__(self):
        self.next: Node | None = None
        self.prev: Node | None = None

class Node(LinkedListNode):
    """
    Base class for network nodes. Inherits from LinkedListNode

    Attributes
    ----------
    label : int
        integer label of the vertex
    old_color : int
        index of the node's current color class
    new_color : int
        index of the node's new color class
    neighbors : list(int)
        list of integers corresponding to the node's neighbors (in the case of DiGraphs, this is their out-edge neighbors, or "successors")
    out_edge_count : int
        Number of out-edges from node to the current ColorClass.
        Used in ColorClass().ComputeStructureSet().
        Needs to be set to zero at the begining of each call to ComputeStructureSet.
    """
    __slots__ = 'label', 'old_color', 'new_color', 'neighbors', \
        'out_edge_count'

    def __init__(self, label: Any, color_class_ind: int, neighbors: List[int]):
        """
        Initialize the node with its label and initial color.
        """
        super().__init__()
        self.label = label

        self.old_color = color_class_ind
        self.new_color = color_class_ind

        if neighbors is None:
            neighbors = []

        self.neighbors = neighbors
        self.out_edge_count = 0

    # magic methods
    def __hash__(self) -> int:
        if type(self.label) == int:
            return self.label
        return self.label.__hash__()

    def __str__(self) -> str:
        return str(self.label)

    
class LinkedList:
    """Base doubly-linked list class"""

    __slots__ = 'head', 'tail', 'size'

    def __init__(self, data: List[Node]=[]):
        """
        Initialize doubly-linked list.

        Parameters
        ----------
        data    : a list of nodes from which to create the doubly-linked list. If 
                    data is not given, initializes an empty linked list.
        """

        if data != []:
            self.head = data[0]
            self.tail = data[-1]
            
            if len(data) > 1:
                self.head.next = data[1]
                self.tail.prev = data[-2]

            self.size = len(data)

            for i in range(1, self.size - 1):
                data[i].prev = data[i - 1]
                data[i].next = data[i + 1]
        else:
            self.head = None
            self.tail = None
            self.size = 0

    def append(self, node: Node):
        """Appends `node` to the list"""

        if self.head is None:
            self.head = node
            self.tail = node
        else:
            if self.tail is None:
                raise ValueError("List is corrupted: tail is None but head is not.")
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        node.next = None  # drop reference to next

    def remove(self, node: Node):
        """
        Removes `node` from list. Assumes `node` is an element of the list,
        however, an error will *not* be raised if it is not.
        """

        if self.head is None:
            raise ValueError("List is empty.")
        
        if node.prev is not None:
            node.prev.next = node.next
        else:  # if node is head
            self.head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:  # if node is tail
            self.tail = node.prev
        node.prev, node.next = None, None  # drop references to next and prev

    # magic methods
    def __list__(self):
        result = list()
        n = self.head

        while n is not None:
            result.append(n)
            n = n.next
        return result
    
    
class ColorClass(LinkedList):
    """
    Base class representing a color class.
    Attributes
    ----------

    """
    __slots__ = 'out_edge_neighbors', 'curr_color_neighbors', \
        'split_color', 'curr_conns', 'max_out_edge_count'

    def __init__(self) -> None:
        """
        Initializes a ColorClass object.
        """
        super().__init__([])
        self.out_edge_neighbors = list()
        self.curr_color_neighbors = 0

    def relabel(self, c):
        """Relabels the v.new_color/v.old_color values of every node v in the linked list"""
        v = self.head
        if v is None:
            raise ValueError("Color Class has no Nodes and therefore cannot be relabeled.")

        while v is not None:
            v.new_color = c
            v.old_color = c
            v = v.next

    def nodes(self) -> List[Any]:
        """Returns a list of node labels for nodes in this ColorClass"""
        labels = list()
        node = self.head

        while node is not None:
            labels.append(node.label)
            node = node.next
        return labels

    def computeColorStructure(self, C: List['ColorClass'], N: List[Node]) -> None:
        """
        Computes the number of edges and color of neighbors of each node in this 
            color class. These metrics are used in splitColor to determine which 
            vertices should be separated into their own color class.

        Parameters
        ----------
        C   : the list of ColorClasses
        N   : the node dictionary

        Complexity
        ----------
        Time: Linear with number of edges in this color class
        Space: Linear with number of neighboring nodes to this class; 
            potentially up to number of edges in this color class, but never 
            worse than all nodes in the graph

        """

        # reset neighbor list
        self.out_edge_neighbors = list()
        self.max_out_edge_count = 0

        w = self.head # first node in the color class
        while w is not None:
            # loop over each out-edge neighbor v of w 
            #   (i.e., there exists an edge v <-- w)
            for v_ind in w.neighbors:
                v = N[v_ind]                                    # get node object
                if v.out_edge_count == 0:                       # check if node v has already been seen
                    C[v.new_color].curr_color_neighbors += 1    # if not, increment out-edge neighbor count for its color
                    self.out_edge_neighbors.append(v)           # and add v to this color class's out-edge neighbors
                v.out_edge_count += 1                           # increment count of out-edges from this color class to v
                # track largest number of outgoing edges to a single node (for bucket sorting)
                if v.out_edge_count > self.max_out_edge_count:
                    self.max_out_edge_count = v.out_edge_count

            w = w.next # move to next node in the color class

    def splitColor(self, C: List['ColorClass'], L: Set[Node]) -> None:
        """
        Uses metrics collected in computeColorStructure to determine which nodes 
            must be moved to a new color class; new color classes are assigned 
            for such nodes (i.e., the new_color attribute is set), but the 
            ColorClass list is not yet changed.

        Parameters
        ----------
        C   : the list of ColorClasses
        L   : the set of nodes that will get new colors

        Complexity
        ----------
        Time: Log-linear (n log(n)) with the number of neighboring nodes
        Space: Linear with number of neighboring nodes

        """
        

        # sort out-edge neighbors by number of edges connecting them to this color class, ascending
        bucketSort(self.out_edge_neighbors, 'out_edge_count', 1, self.max_out_edge_count)
        # NOTE: this may actually be somewhat slower in practice for many graphs, but is necessary 
        #   guarantee linear sorting complexity and m log(n) complexity overall. The alternative:
        # self.out_edge_neighbors.sort(key=operator.attrgetter('out_edge_count'))

        visited = set() # which ColorClasses have been visited
        for v in self.out_edge_neighbors:
            # new_color may have been changed in previous iterations, so we may 
            #   not use old_color here
            if v.new_color not in visited:
                visited.add(v.new_color)
                b = v.new_color
                # set curr_conns to the smallest number of connections that a 
                #   node in C[b] has with this color class
                if C[b].curr_color_neighbors < C[b].size:
                    # if not all nodes in C[b] neighbor a node in this color class, 
                    #   then the min number of connections to this color class is zero
                    C[b].curr_conns = 0
                else:
                    # otherwise, the minimum number of connections is v.out_edge_count
                    #   (since out_edge_neighbors was sorted by out_edge_count)
                    C[b].curr_conns = v.out_edge_count

                C[b].split_color = b # initialize split_color for use in next loop
                C[b].curr_color_neighbors = 0 # resetting count for the next iteration

        for v in self.out_edge_neighbors:
            b = v.new_color
            # curr_conns is the min number of connections in C[b] to the current color class. Nodes 
            #   with more than this number of connections get moved into a different color class.
            # Note that since out_edge_neighbors were sorted by out_edge_count, we will exhaust all nodes
            #   with an out_edge_count of curr_conns before moving on to nodes with larger out_edge_counts
            if C[b].curr_conns != v.out_edge_count:
                C[b].curr_conns = v.out_edge_count   # update curr_conns with the new out_edge_count
                C.append(ColorClass())               # add new color
                C[b].split_color = len(C) - 1        # update split to apply to subsequent nodes

            # As soon as we have processed all nodes v from C[b] with minimum out_edge_count, the 
            #   split_color of C[b] will change (in the above if statement). All subsequent nodes 
            #   from C[b] will recieve new_color values according to their out_edge_count (thus, all 
            #   v in C[b] with equal out_edge_count will be given the same new_color value). The only 
            #   nodes that will retain their original color value will be the nodes from each C[b] 
            #   with the same minimum out_edge_count
            if v.new_color != C[b].split_color:   # if split_color of C[b] changed
                L.add(v)
                
                # NOTE: it may seem more intuitive to update the ColorClass sizes when nodes are 
                #   added or removed (in recolor); HOWEVER, we use the updated sizes before that 
                #   point (e.g., future iterations of splitColor before recolor is called).
                C[v.new_color].size -= 1
                v.new_color = C[b].split_color
                C[v.new_color].size += 1

    # magic methods
    def __str__(self):
        v = self.head
        if v is None:
            return 'None'

        data = f'{v}'
        while True:
            if v.next is None:
                return data

            v = v.next
            data += f', {v}'


def bucketSort(objs: List[Any], attribute: str, attr_min_val: int, attr_max_val: int) -> None:
    """
    Performs an in-place bucket sort on the list of objects `objs` based on the specified `attribute`.

    Parameters
    ----------
    objs        : the list to be sorted
    attribute   : the attribute by which to sort
    attr_min_val: the smallest possible value of `attribute`
    attr_max_val: the largest possible value of `attribute`

    Complexity
    ----------
    Time: Linear with length of objs and (attr_max_val - attr_min_val)
    Space: Linear with length of objs and (attr_max_val - attr_min_val)

    """
    buckets: List[List[Any]] = [[] for _ in range(attr_min_val, attr_max_val + 1)]
    for obj in objs:
        index = getattr(obj, attribute) - attr_min_val
        buckets[index].append(obj)
    i = 0
    for bucket in buckets:
        for obj in bucket:
            objs[i] = obj
            i += 1
   

def initFromNx(G: nx.Graph | nx.DiGraph) -> List[Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    G   : the graph to be analyzed

    Returns
    -------
    N   : a list of Node objects representing the nodes of G

    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    # initialize Node list -- all start with ColorClass index of 0
    # in DiGraphs, neighbors() is the same as successors()
    N = [Node(node, 0, list(G.neighbors(node))) for node in G.nodes()]

    return N


def initFromSparse(mat: sp.csr_array) -> List[Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    G   : the graph to be analyzed

    Returns
    -------
    N   : a list of Node objects representing the nodes of G

    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    # initialize Node list -- all start with ColorClass index of 0
    N = [Node(i, 0, list(mat.indices[mat.indptr[i]:mat.indptr[i + 1]])) for i in range(mat.shape[0])]

    return N


def recolor(C: List[ColorClass], L: Set[Node]) -> None:
    """
    Updates color classes to reflect the coloring stored in each node's 
        new_color attribute. When a color class splits, the largest derived 
        color class keeps the original color.
    
    Parameters
    ----------
    C   : the list of ColorClasses
    L   : the set of nodes that will get new colors

    Complexity
    ----------
    Time: Linear with len(L)
    Space: Linear with len(L)

    """

    for v in L:
        C[v.old_color].remove(v)
        C[v.new_color].append(v)

    # make sure largest new color retains old color label (for a more efficient next iteration)
    for c in {v.old_color for v in L}:
        # get index of largest new colorclass from same previous colorclass
        d = max({(C[v.new_color].size, v.new_color) for v in L if v.old_color == c})[1]
        # if color d has more nodes than the original, switch their coloring
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    for v in L:
        v.old_color = v.new_color

def getEquitablePartition(N: List[Node], verbose_progress: bool=False) -> Dict[int, List[Any]]:
    """
    Finds the coarsest equitable partition of a network (receiving equitable partition if directed).
     In the case of a directed graph, it computes the coarsest receiving equitable partition.
    
    Parameters
    ----------
    N   : a list of Node objects representing the nodes of the graph to be 
            analyzed
    
    Returns
    -------
    ep  : a dictionary of (int, list) where each list represents nodes in the 
            same partition element

    Complexity
    ----------
    Time: O(m log(n)), where m is the number of edges and n is the number of nodes
    Space: O(n)

    """

    # initialize ColorClass list
    C = [ColorClass()]
    
    # add all nodes to the first color class to start
    for n in N:
        C[0].append(n)

    C[0].size = len(N)

    iters = 0
    if verbose_progress:
        print("Finding Coarsest EP...")
    
    prev_color_count = 0 # number of colors in the previous iteration

    # NOTE: the complexity of each iteration is proportional to 
    #   SUM from i=0 -> len(L) of degree(L[i]).
    #   Hence, if every node is recolored once, the complexity is bounded by the 
    #   total number of edges, m. Each node may be recolored at most log(n) 
    #   times, where n is the number of nodes. Hence, we have an overall 
    #   complexity of O(m log(n)).

    while len(C) > prev_color_count:
        L = set() # nodes with new colors

        color_count = len(C)

        # iterate over newly created colors from previous iteration
        for c in range(prev_color_count, len(C)):
            C[c].computeColorStructure(C, N)

            C[c].splitColor(C, L)

            for v in C[c].out_edge_neighbors:
                v.out_edge_count = 0

        recolor(C, L)

        prev_color_count = color_count

        iters += 1
        if verbose_progress:
            updateProgress(iters)
        
    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C))}

    if verbose_progress:
        updateProgress(iters, finished=True)

    return ep

def updateProgress(iterations, finished=False):
    print(f"\r{iterations} iterations completed.", end='')
    if finished:
        print(" EP algorithm complete!")