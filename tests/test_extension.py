import higra_distributed as m
import unittest
import numpy as np
import higra as hg

class ExampleTest(unittest.TestCase):

    @staticmethod
    def get_test_tree():
        """
        base graph is

        (0)-- 0 --(1)-- 2 --(2)
         |         |         |
         6         6         0
         |         |         |
        (3)-- 0 --(4)-- 4 --(5)
         |         |         |
         5         5         3
         |         |         |
        (6)-- 0 --(7)-- 1 --(8)

        Minima are
        A: (0, 1)
        B: (3, 4)
        C: (2, 5)
        D: (6, 7)

        BPT:




        4                 +-------16------+
                          |               |
        3         +-------15-----+        |
                  |              |        |
        2     +---14--+          |        |
              |       |          |        |
        1     |       |       +--13-+     |
              |       |       |     |     |
        0   +-9-+   +-10+   +-12+   |   +-11+
            +   +   +   +   +   +   +   +   +
            0   1   2   5   6   7   8   3   4


        :return:
        """

        g = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))

        return hg.bpt_canonical(g, edge_weights)

    def test_example_function(self):
        tree, altitudes = ExampleTest.get_test_tree()
        
        result = m.example_function(tree, altitudes)

        expected_result = (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 5., 4.)
        self.assertTrue(np.allclose(result, expected_result))

if __name__ == '__main__':
    unittest.main()
