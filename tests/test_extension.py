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

        return hg.bpt_canonical(g, edge_weights)

    def test_select(self):
        base_tree, altitudes = ExampleTest.get_test_tree()

        tree, node_map = m.select(base_tree, (0, 1, 6, 7))
        ref_parents = (4, 4, 5, 5, 6, 7, 7, 7)
        ref_node_map = (0, 1, 6, 7, 14, 17, 21, 22)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))

        tree, node_map = m.select(base_tree, (2, 3, 8, 9))
        ref_parents = (4, 4, 5, 5, 6, 7, 7, 8, 9, 9)
        ref_node_map = (2, 3, 8, 9, 13, 15, 19, 20, 21, 22)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))

        tree, node_map = m.select(base_tree, (10, 5, 11, 4))
        ref_parents = (4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10)
        ref_node_map = (4, 5, 10, 11, 12, 16, 18, 19, 20, 21, 22)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))

    def test_join(self):
        tree1 = hg.Tree((2, 3, 4, 4, 5, 6, 6))
        node_map1 = np.asarray((3, 9, 4, 13, 7, 2, 3))
        mst_edge_weights1 = np.asarray((2, 4, 11, 12, 15))

        tree2 = hg.Tree((2, 3, 4, 4, 4))
        node_map2 = np.asarray((4, 10, 8, 15, 10))
        mst_edge_weights2 = np.asarray((1, 5, 7))

        border_edge_sources = np.asarray((0, 1))
        border_edge_targets = np.asarray((0, 1))
        border_edge_map = np.asarray((6, 14))
        border_edge_weights = np.asarray((8, 9))

        tree, node_map = m.join(tree1, node_map1, mst_edge_weights1,
                                tree2, node_map2, mst_edge_weights2,
                                border_edge_sources, border_edge_targets,
                                border_edge_map, border_edge_weights)

        ref_parents = (5, 6, 4, 7, 8, 9, 10, 8, 9, 10, 11, 12, 12)
        ref_node_map = (3, 9, 4, 10, 8, 4, 13, 15, 10, 6, 14, 2, 3)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))

    def test_insert(self):
        tree1 = hg.Tree((2, 3, 4, 4, 5, 6, 7, 8, 8))
        node_map1 = np.asarray((4, 10, 8, 15, 10, 6, 14, 2, 3))
        mst_weights1 = np.asarray((1, 5, 7, 8, 9, 12, 15))
        tree1_2_tree2_leaves_map = np.asarray((0, 2))

        tree2 = hg.Tree((4, 4, 5, 5, 6, 6, 6))
        node_map2 = np.asarray((4, 5, 10, 11, 8, 15, 10))
        mst_weights2 = np.asarray((1, 5, 7))

        tree, node_map = m.insert(tree1, node_map1, mst_weights1,
                                  tree2, node_map2, mst_weights2,
                                  tree1_2_tree2_leaves_map)

        ref_parents = (4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 10)
        ref_node_map = (4, 5, 10, 11, 8, 15, 10, 6, 14, 2, 3)
        self.assertTrue(np.all(tree.parents() == ref_parents))
        self.assertTrue(np.all(node_map == ref_node_map))


if __name__ == '__main__':
    unittest.main()
