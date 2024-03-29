############################################################################
# Copyright ESIEE Paris (2021)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra_distributed as m
import unittest
import numpy as np
import higra as hg


class ExampleTest(unittest.TestCase):

    @staticmethod
    def get_test_tree():
        """
        base graph is

        (0)-- 3 --(1)-- 12--(2)-- 2 --(3)-- 8 --(4)-- 1 --(5)
         |         |         |         |         |         |
         16        15        14        11        10        7
         |         |         |         |         |         |
        (6)-- 6 --(7)-- 13--(8)-- 4 --(9)-- 9 --(10)- 5 --(11)




        BPT:
                                        +------22-----+
                                        |             |
                               +--------21----+       |
                               |              |       |
                        +------20-----+       |       |
                        |             |       |       |
                  +-----19----+       |       |       |
                  |           |       |       |       |
              +---18--+       |       |       |       |
              |       |       |       |       |       |
            +-12+   +-16+   +-13+   +-15+   +-14+   +-17+
            +   +   +   +   +   +   +   +   +   +   +   +
            4   5   10  11  2   3   8   9   0   1   6   7


        :return:
        """

        g = hg.get_4_adjacency_graph((2, 6))
        edge_weights = np.asarray((3, 16, 12, 15, 2, 14, 8, 11, 1, 10, 7, 6, 13, 4, 9, 5))
        tree, altitudes = hg.bpt_canonical(g, edge_weights)

        tree_mst_edge_map = tree.mst_edge_map
        leaf_map = np.arange(g.num_vertices())
        mst_edge_map = tree_mst_edge_map

        node_map = np.concatenate((leaf_map, mst_edge_map))

        tree.node_map = node_map
        tree.mst_weights = edge_weights[tree_mst_edge_map]

        return tree

    def test_select(self):
        base_tree = ExampleTest.get_test_tree()

        tree, node_map, mst_weights = m.cpp._select(base_tree, base_tree.node_map, base_tree.mst_weights, (0, 1, 6, 7))
        ref_parents = (4, 4, 5, 5, 6, 7, 7, 7)
        ref_node_map = (0, 1, 6, 7, 0, 11, 2, 12)
        ref_mst_weights = (3, 6, 12, 13)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

        tree, node_map, mst_weights = m.cpp._select(base_tree, base_tree.node_map, base_tree.mst_weights, (2, 3, 8, 9))
        ref_parents = (4, 4, 5, 5, 6, 7, 7, 8, 9, 9)
        ref_node_map = (2, 3, 8, 9, 4, 13, 6, 14, 2, 12)
        ref_mst_weights = (2, 4, 8, 9, 12, 13)

        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

        tree, node_map, mst_weights = m.cpp._select(base_tree, base_tree.node_map, base_tree.mst_weights,
                                                    (4, 5, 10, 11))
        ref_parents = (4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10)
        ref_node_map = (4, 5, 10, 11, 8, 15, 10, 6, 14, 2, 12)
        ref_mst_weights = (1, 5, 7, 8, 9, 12, 13)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

    def test_join(self):
        tree1 = hg.Tree((2, 3, 4, 4, 5, 6, 6))
        node_map1 = np.asarray((3, 9, 4, 13, 7, 2, 3))
        mst_edge_weights1 = np.asarray((2, 4, 11, 12, 15))

        tree2 = hg.Tree((2, 3, 4, 4, 4))
        node_map2 = np.asarray((4, 10, 8, 15, 10))
        mst_edge_weights2 = np.asarray((1, 5, 7))

        border_edge_sources = np.asarray((3, 9))
        border_edge_targets = np.asarray((4, 10))
        border_edge_map = np.asarray((6, 14))
        border_edge_weights = np.asarray((8, 9))

        tree, node_map, mst_weights = m.cpp._join(tree1, node_map1, mst_edge_weights1,
                                                  tree2, node_map2, mst_edge_weights2,
                                                  border_edge_sources, border_edge_targets,
                                                  border_edge_map, border_edge_weights)

        ref_parents = (5, 6, 4, 7, 8, 9, 10, 8, 9, 10, 11, 12, 12)
        ref_node_map = (3, 9, 4, 10, 8, 4, 13, 15, 10, 6, 14, 2, 3)
        ref_mst_weights = (1, 2, 4, 5, 7, 8, 9, 12, 15)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

    def test_join_bad_edge_order(self):
        tree1 = hg.Tree((2, 3, 4, 4, 5, 6, 6))
        node_map1 = np.asarray((3, 9, 4, 13, 7, 2, 3))
        mst_edge_weights1 = np.asarray((2, 4, 11, 12, 15))

        tree2 = hg.Tree((2, 3, 4, 4, 4))
        node_map2 = np.asarray((4, 10, 8, 15, 10))
        mst_edge_weights2 = np.asarray((1, 5, 7))

        border_edge_sources = np.asarray((9, 3))
        border_edge_targets = np.asarray((10, 4))
        border_edge_map = np.asarray((14, 6))
        border_edge_weights = np.asarray((9, 8))

        tree, node_map, mst_weights = m.cpp._join(tree1, node_map1, mst_edge_weights1,
                                                  tree2, node_map2, mst_edge_weights2,
                                                  border_edge_sources, border_edge_targets,
                                                  border_edge_map, border_edge_weights)

        ref_parents = (5, 6, 4, 7, 8, 9, 10, 8, 9, 10, 11, 12, 12)
        ref_node_map = (3, 9, 4, 10, 8, 4, 13, 15, 10, 6, 14, 2, 3)
        ref_mst_weights = (1, 2, 4, 5, 7, 8, 9, 12, 15)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

    def test_insert(self):
        tree1 = hg.Tree((2, 3, 4, 4, 5, 6, 7, 8, 8))
        node_map1 = np.asarray((4, 10, 8, 15, 10, 6, 14, 2, 3))
        mst_weights1 = np.asarray((1, 5, 7, 8, 9, 12, 15))

        tree2 = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        node_map2 = np.asarray((4, 5, 10, 11, 8, 15, 10))
        mst_weights2 = np.asarray((1, 5, 7))

        tree, node_map, mst_weights = m.cpp._insert(tree1, node_map1, mst_weights1,
                                                    tree2, node_map2, mst_weights2)

        ref_parents = (4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10)
        ref_node_map = (4, 5, 10, 11, 8, 15, 10, 6, 14, 2, 3)
        ref_mst_weights = (1, 5, 7, 8, 9, 12, 15)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))
        self.assertTrue(np.all(mst_weights == ref_mst_weights))

    def test_distributed_bpt(self):
        size = 100
        slice_size = 13
        np.random.seed(42)
        image = np.random.randint(0, 256, (size, size))

        causal_graph = m.CausalGraph_2d_4adj(image, slice_size)
        distributed_hierarchy = m.distributed_canonical_bpt(causal_graph)

        causal_graph_trivial = m.CausalGraph_2d_4adj(image, size)
        ref_tree = m.partial_bpt(causal_graph_trivial, 0)

        for i in range(causal_graph.num_slices):
            ref_tree_i = m.select(ref_tree, causal_graph.get_vertex_map(i))

            self.assertTrue(np.all(ref_tree_i.parents() == distributed_hierarchy[i].parents()))
            self.assertTrue(np.all(ref_tree_i.node_map == distributed_hierarchy[i].node_map))
            self.assertTrue(np.all(ref_tree_i.mst_weights == distributed_hierarchy[i].mst_weights))


if __name__ == '__main__':
    unittest.main()
