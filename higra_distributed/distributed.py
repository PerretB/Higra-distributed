from . import higra_distributedm as cpp
import higra as hg
import numpy as np
import math

"""
Vocabulary:

- graph: the complete graph representing the whole image. This graph never has to be in memory at once. 
  We assume that we have a bijection between the vertices (resp. the edges) of the graph and the set of integers 
  |[0, n |[ where n is equal to the number of vertices (resp. edges in the graph). The edges of this graph are weighted.

- distributed tree: a tree (:class:`~higra.Tree`), with 2 particular attributes:
    - node_map: a 1d array of dtype int64 and of size tree.num_vertices() such that, for any node index i, 
      node_map[i] is the index of the corresponding vertex in the full graph if i is a leaf node, and
      node_map[i] is the index of the corresponding mst edge in the full graph if i is an internal node.
    - mst_weights: : a 1d array of dtype float64 of size tree.num_vertices() - tree.num_leaves(), such that, 
      for any internal node index i, mst_weights[i - tree.num_leaves()] is the weight of the mst edge corresponding
      to the node i in the full graph

"""


def select(tree, selected_leaves):
    """
    Select a sub distributed tree of the given distributed tree containing the given set of leaf nodes.

    :param tree: a distributed tree (:class:`~higra.Tree`)
    :param selected_leaves: 1d array containing the indices of the selected leaf nodes (nd array of dtype int64)
    :return: a new distributed tree
    """
    new_tree, new_node_map = cpp._select(tree, selected_leaves)

    new_tree.node_map = tree.node_map[new_node_map]
    new_tree.mst_weights = tree.mst_weights[new_node_map[new_tree.num_leaves():] - tree.num_leaves()]
    return new_tree


def join(tree1, tree2, weighted_border):
    """
    Join two distributed trees :attr:`tree1` and :attr:`tree2` through their common weighted border :attr:`weighted_border`.

    A weighted border represent all the edges of the full graph joining a leaf node of :attr:`tree1` to a leaf node of
    :attr:`tree2`. If there is n such edges, :attr:`weighted_border` is a tuple of four elements:

        - a 1d array *border_edge_sources* of dtype int64 of size n, the source edge vertices in tree 1: for all i,
          border_edge_sources[i] is the vertex index of :attr:`tree1` contained in the i-th edge of the border
        - a 1d array *border_edge_targets* of dtype int64 of size n, the target edge vertices in tree 2: for all i,
          border_edge_targets[i] is the vertex index of :attr:`tree2` contained in the i-th edge of the border
        - a 1d array *border_edge_map* of dtype int64 of size n, the edge index in the full graph: for all i,
          border_edge_map[i] is the index of the i-th edge in the full graph
        - a 1d array *border_edge_weights* of dtype float64 of size n, the weight of the edge in the full graph:
          for all i, border_edge_weights[i] is the weight of the i-th edge in the full graph

    :param tree1: first distributed tree (:class:`~higra.Tree`)
    :param tree2: second distributed tree (:class:`~higra.Tree`)
    :param weighted_border: edges joining leaf nodes of tree1 to leaf nodes of tree2
    :return: a new distributed tree
    """
    border_edge_sources, border_edge_targets, border_edge_map, border_edge_weights = weighted_border
    tree, node_map, mst_weights = cpp._join(tree1, tree1.node_map, tree1.mst_weights,
                                            tree2, tree2.node_map, tree2.mst_weights,
                                            border_edge_sources, border_edge_targets,
                                            border_edge_map, border_edge_weights)
    tree.node_map = node_map
    tree.mst_weights = mst_weights
    return tree


def insert(tree1, tree2):
    """
    Insert the distributed tree :attr:`tree1` into the distributed tree :attr:`tree2`.

    :param tree1: the distributed tree to be inserted (:class:`~higra.Tree`)
    :param tree2: the distributed tree receiving the insertion  (:class:`~higra.Tree`)
    :return: a new distributed tree
    """
    tree, node_map, mst_weights = cpp._insert(tree1, tree1.node_map, tree1.mst_weights,
                                              tree2, tree2.node_map, tree2.mst_weights)

    tree.node_map = node_map
    tree.mst_weights = mst_weights
    return tree


class causal_4adj_graph:
    """
    A causal 4 adjacency graph, sliced into bands composed of a given number of lines.
    More precisely a slice of the graph is a continuous sequence of vertices of the graph starting at the beginning of
    a line *l* and stopping at the end of a line *l' >= l*.
    Slices are numbered from the top (0) to the bottom (self.num_slices) of the graph.

    Each slice represent a sub-graph of the complete graph and each slice is adjacent to at most 2 other slices (one
    above and one bellow).
    """
    def __init__(self, image, max_slice_size, weight_function=lambda x, y: np.abs(x - y)):
        self.image = image
        self.max_slice_size = max_slice_size
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.num_slices = int(math.ceil(self.height / max_slice_size))
        self.weight_function = weight_function
        self._num_h_edges = (self.width - 1) * self.height

    def _get_slice_interval(self, slice_number):
        """
        Assume that the slice :attr:`slice_number` spans from the line *start* (included) to the line *stop* (excluded),
        then this function returns the pair *(start, stop)*.

        For all the slices, except perhaps the last one, we have *stop - start* equals to *self.max_slice_size*

        :param slice_number: positive integer strictly smaller than self.num_slices
        :return: a pair of line numbers (start, stop)
        """
        assert 0 <= slice_number < self.num_slices, "Invalid slice number"
        return slice_number * self.max_slice_size, min((slice_number + 1) * self.max_slice_size, self.height)

    def get_vertex_map(self, slice_number):
        """
        Return a 1d array of dtype int64 containing the indices, in increasing order,
        of the vertices contained in the *slice_number*-th slice of the causal graph.

        :param slice_number: positive integer strictly smaller than self.num_slices
        :return:
        """
        start, stop = self._get_slice_interval(slice_number)
        return np.arange(start * self.width, stop * self.width)

    def get_edge_map(self, slice_number):
        """
        Return a 1d arrays of dtype int64 containing the indices, in increasing order,
        of the edge contained in the *slice_number*-th slice of the causal graph.

        :param slice_number:
        :return:
        """
        start, stop = self._get_slice_interval(slice_number)
        h_edges = np.arange(start * (self.width - 1), stop * (self.width - 1))
        v_edges = np.arange(self._num_h_edges + start * self.width, self._num_h_edges + (stop - 1) * self.width)
        return np.concatenate((h_edges, v_edges))

    def get_edge_weighted_graph(self, slice_number):
        """
        Return an edge weighted graph, ie a pair (graph, edge_weights) representing the given slice of the graph.

        The graph is given as a triplet:

           - a 1d array *sources* of dtype int64 of size n (the number of edges in the sub-graph), representing the source
             vertices of the edges of the graph: for all i, *sources[i]* is the first vertex of the i-th edge
           - a 1d array *targets* of dtype int64 of size n (the number of edges in the sub-graph), representing the target
             vertices of the edges of the graph: for all i, *targets[i]* is the second vertex of the i-th edge
           - a positive integer: the number of vertices in the graph

        The vertices and the edges of the sub-graph corresponding to the slice can be mapped to the vertices and
        edges of the full graph thanks to the method get_vertex_map and get_edge_map

        The edge weights are represented by a 1d array of dtype float64 and of size n (the number of edges in the sub-graph):
        for all i, *edge_weights[i]* is the weight of the i-th edge

        :param slice_number:
        :return: graph, edge_weights
        """
        start, stop = self._get_slice_interval(slice_number)
        shape = (stop - start, self.width)
        num_vertices = shape[0] * shape[1]
        vertices = np.arange(num_vertices).reshape(shape)
        h_edges_sources, h_edges_targets = vertices[:, :-1].ravel(), vertices[:, 1:].ravel()
        v_edges_sources, v_edges_targets = vertices[:-1, :].ravel(), vertices[1:, :].ravel()

        edges_sources = np.concatenate((h_edges_sources, v_edges_sources))
        edges_targets = np.concatenate((h_edges_targets, v_edges_targets))

        im = self.image[start:stop, :].ravel()
        edge_weights = self.weight_function(im[edges_sources], im[edges_targets])

        return (edges_sources, edges_targets, num_vertices), edge_weights

    def get_slice_back_frontier(self, slice_number):
        """
        Get the vertex index of the given slice that are on the frontier

        :param slice_number:
        :return:
        """
        assert 0 < slice_number < self.num_slices, "Invalid slice number"
        return np.arange(self.width)

    def get_slice_front_frontier(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        return np.arange(self.width * (self.max_slice_size - 1), self.width * self.max_slice_size)

    def get_weighted_border_edges(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        start, stop = self._get_slice_interval(slice_number)

        im = self.image[stop - 1:stop + 1, :]
        edge_sources = np.arange(self.width)
        edge_targets = edge_sources
        edge_map = np.arange(self._num_h_edges + (stop - 1) * self.width, self._num_h_edges + stop * self.width)
        edge_weights = self.weight_function(im[0, :].ravel(), im[1, :].ravel())

        return edge_sources, edge_targets, edge_map, edge_weights

    def get_border_back(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        return np.arange(self.width)

    def get_border_front(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        return np.arange(self.width, 2 * self.width)


def partial_bpt(causal_graph, slice_number):
    graph, edge_weights = causal_graph.get_edge_weighted_graph(slice_number)
    tree = hg.bpt_canonical(graph, edge_weights, return_altitudes=False, compute_mst=False)
    tree_mst_edge_map = tree.mst_edge_map
    leaf_map = causal_graph.get_vertex_map(slice_number)
    mst_edge_map = causal_graph.get_edge_map(slice_number)[tree_mst_edge_map]

    node_map = np.concatenate((leaf_map, mst_edge_map))
    tree.node_map = node_map
    tree.mst_weights = edge_weights[tree_mst_edge_map]
    return tree


def distributed_canonical_bpt(causal_graph):
    b_up = [partial_bpt(causal_graph, 0)]
    m_up = [None]
    m_down = [None] * causal_graph.num_slices
    b_down = [None] * causal_graph.num_slices
    for i in range(causal_graph.num_slices - 1):
        bpt_ip1 = partial_bpt(causal_graph, i + 1)

        b_i_border_tree = select(b_up[i], causal_graph.get_slice_front_frontier(i))
        b_ip1_border_tree = select(bpt_ip1, causal_graph.get_slice_back_frontier(i + 1))
        m_up_ip1 = join(b_i_border_tree, b_ip1_border_tree, causal_graph.get_weighted_border_edges(i))

        b_up_ip1 = insert(select(m_up_ip1, causal_graph.get_border_front(i)), bpt_ip1)
        m_up.append(m_up_ip1)
        b_up.append(b_up_ip1)

    m_down[-1] = m_up[-1]
    b_down[-1] = b_up[-1]

    for i in range(causal_graph.num_slices - 2, -1, -1):
        b_down[i] = insert(select(m_down[i + 1], causal_graph.get_border_back(i)), b_up[i])
        if i > 0:
            m_down[i] = insert(select(b_down[i], causal_graph.get_slice_back_frontier(i)), m_up[i])

    return b_down
