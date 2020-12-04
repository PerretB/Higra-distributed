from . import higra_distributedm as cpp


def select(tree, selected_leaves):
    return cpp._select(tree, selected_leaves)


def join(tree1, node_map1, mst_weights1,
         tree2, node_map2, mst_weights2,
         border_edge_sources, border_edge_targets,
         border_edge_map, border_edge_weights):
    return cpp._join(tree1, node_map1, mst_weights1,
                    tree2, node_map2, mst_weights2,
                    border_edge_sources, border_edge_targets,
                    border_edge_map, border_edge_weights)


def insert(tree1, node_map1, mst_weights1,
           tree2, node_map2, mst_weights2,
           tree1_2_tree2_leaves_map):
    return cpp._insert(tree1, node_map1, mst_weights1,
                      tree2, node_map2, mst_weights2,
                      tree1_2_tree2_leaves_map)
