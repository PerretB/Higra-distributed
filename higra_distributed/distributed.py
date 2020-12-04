from . import higra_distributedm as cpp
import higra as hg
import numpy as np
import math


def select(tree, selected_leaves):
    new_tree, new_node_map = cpp._select(tree, selected_leaves)

    new_tree.node_map = tree.node_map[new_node_map]
    new_tree.mst_weights = tree.mst_weights[new_node_map[new_tree.num_leaves():] - tree.num_leaves()]
    return new_tree


def join(tree1, tree2, weighted_border):
    border_edge_sources, border_edge_targets, border_edge_map, border_edge_weights = weighted_border
    tree, node_map, mst_weights = cpp._join(tree1, tree1.node_map, tree1.mst_weights,
                                            tree2, tree2.node_map, tree2.mst_weights,
                                            border_edge_sources, border_edge_targets,
                                            border_edge_map, border_edge_weights)
    tree.node_map = node_map
    tree.mst_weights = mst_weights
    return tree


def insert(tree1, tree2):
    tree, node_map, mst_weights = cpp._insert(tree1, tree1.node_map, tree1.mst_weights,
                                              tree2, tree2.node_map, tree2.mst_weights)

    tree.node_map = node_map
    tree.mst_weights = mst_weights
    return tree


class causal_4adj_graph:
    def __init__(self, image, max_slice_size, weight_function=lambda x, y: np.abs(x - y)):
        self.image = image
        self.max_slice_size = max_slice_size
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.num_slices = int(math.ceil(self.height / max_slice_size))
        self.weight_function = weight_function
        self._num_h_edges = (self.width - 1) * self.height

    def get_slice_interval(self, slice_number):
        assert 0 <= slice_number < self.num_slices, "Invalid slice number"
        return slice_number * self.max_slice_size, min((slice_number + 1) * self.max_slice_size, self.height)

    def get_node_map(self, slice_number):
        start, stop = self.get_slice_interval(slice_number)
        return np.arange(start * self.width, stop * self.width)

    def get_edge_map(self, slice_number):
        start, stop = self.get_slice_interval(slice_number)
        h_edges = np.arange(start * (self.width - 1), stop * (self.width - 1))
        v_edges = np.arange(self._num_h_edges + start * self.width, self._num_h_edges + (stop - 1) * self.width)
        return np.concatenate((h_edges, v_edges))

    def get_edge_weighted_graph(self, slice_number):
        start, stop = self.get_slice_interval(slice_number)
        g = hg.UndirectedGraph((stop - start) * self.width)
        shape = (stop - start, self.width)
        hg.CptGridGraph.link(g, (stop - start, self.width))
        vertices = np.arange(g.num_vertices()).reshape(shape)
        h_edges_sources, h_edges_targets = vertices[:, :-1].ravel(), vertices[:, 1:].ravel()
        v_edges_sources, v_edges_targets = vertices[:-1, :].ravel(), vertices[1:, :].ravel()

        edges_sources = np.concatenate((h_edges_sources, v_edges_sources))
        edges_targets = np.concatenate((h_edges_targets, v_edges_targets))
        g.add_edges(edges_sources, edges_targets)

        im = self.image[start:stop, :].ravel()
        edge_weights = self.weight_function(im[edges_sources], im[edges_targets])

        return g, edge_weights

    def get_slice_back_frontier(self, slice_number):
        assert 0 < slice_number < self.num_slices, "Invalid slice number"
        return np.arange(self.width)

    def get_slice_front_frontier(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        return np.arange(self.width * (self.max_slice_size - 1), self.width * self.max_slice_size)

    def get_weighted_border_edges(self, slice_number):
        assert 0 <= slice_number < self.num_slices - 1, "Invalid slice number"
        start, stop = self.get_slice_interval(slice_number)

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
    hg.clear_attributes(tree)
    tree.leaf_graph = None
    tree_mst_edge_map = tree.mst_edge_map
    leaf_map = causal_graph.get_node_map(slice_number)
    mst_edge_map = causal_graph.get_edge_map(slice_number)[tree_mst_edge_map]

    node_map = np.concatenate((leaf_map, mst_edge_map))
    tree.node_map = node_map
    tree.mst_weights = edge_weights[tree_mst_edge_map]
    return tree


def distributed_canonical_bpt(causal_graph):
    from guppy import hpy
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
        print("Sizes","b", b_up_ip1.num_vertices())
        print("Sizes","m", m_up_ip1.num_vertices())
        m_up.append(m_up_ip1)
        b_up.append(b_up_ip1)


    print(hpy().heap()[0])

    m_down[-1] = m_up[-1]
    b_down[-1] = b_up[-1]

    for i in range(causal_graph.num_slices - 2, -1, -1):
        b_down[i] = insert(select(m_down[i + 1], causal_graph.get_border_back(i)), b_up[i])
        if i > 0:
            m_down[i] = insert(select(b_down[i], causal_graph.get_slice_back_frontier(i)), m_up[i])

    print(hpy().heap())

    return b_down

def bench():
    from time import time

    size = 500
    slice_size = 50
    np.random.seed(42)
    image = np.random.randint(0, 256, (size, size))
    t1 = time()
    causal_graph = causal_4adj_graph(image, slice_size)
    distributed_hierarchy = distributed_canonical_bpt(causal_graph)
    t2 = time()
    print(t2 - t1)
    return distributed_hierarchy