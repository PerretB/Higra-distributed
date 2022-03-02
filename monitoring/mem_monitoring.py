from tqdm import tqdm
from build import higra_distributed as m
from typing import Tuple

def get_size(t, types: Tuple[int,int,int] = (1,1,1), bytes=False) -> int:
    """
        Get size of a given hierarchy.

    Args:
        t (_type_): Hierarchy
        types (Tuple[int,int,int], optional): Number of bytes used by each element.
         (parents, node_map, weights). Defaults to (1,1,1).
        bytes (bool, optional): If 'bytes' is True, the size of the hierarchy will
         be expressed in bytes. The types of each element will be used for the
         calculation. Defaults to False.

    Returns:
        int: _description_
    """
    c1,c2,c3 = (t.parents().itemsize, t.node_map.itemsize, t.mst_weights.itemsize) if bytes else types
    size = c1*len(t.parents()) + c2*len(t.node_map) + c3*len(t.mst_weights)
    return size

def get_elem_size(causal_graph, disable_tqdm=True):
    s_b_i_border_tree = []
    s_b_ip1_border_tree = []
    s_m_up = []
    s_b_up = []
    s_m_down = []
    s_b_down = []

    m_b_up = []
    m_b_down = []

    b_up = [m.partial_bpt(causal_graph, 0)]
    m_up = [None]
    m_down = [None] * causal_graph.num_slices
    b_down = [None] * causal_graph.num_slices

    s_b_i_border_tree.append(0)
    s_b_ip1_border_tree.append(0)
    s_m_up.append(0)
    s_b_up.append(get_size(b_up[0]))
    m_b_up.append(get_size(b_up[0], bytes=True))

    for i in tqdm(range(causal_graph.num_slices - 1), disable=disable_tqdm):
        bpt_ip1 = m.partial_bpt(causal_graph, i + 1)

        front_slice_i = causal_graph.get_slice_front_frontier(i)
        back_slice_ip1 = causal_graph.get_slice_back_frontier(i + 1)
        b_i_border_tree = m.select(b_up[i], front_slice_i)
        b_ip1_border_tree = m.select(bpt_ip1, back_slice_ip1)

        m_up_ip1 = m.join(b_i_border_tree, b_ip1_border_tree,
                        causal_graph.get_weighted_border_edges(i))

        b_up_ip1 = m.insert(m.select(m_up_ip1, back_slice_ip1), bpt_ip1)
        m_up.append(m_up_ip1)
        b_up.append(b_up_ip1)

        s_b_i_border_tree.append(get_size(b_i_border_tree))
        s_b_ip1_border_tree.append(get_size(b_ip1_border_tree))
        s_m_up.append(get_size(m_up_ip1))
        s_b_up.append(get_size(b_up_ip1))
        m_b_up.append(get_size(b_up_ip1, bytes=True))

    m_down[-1] = m_up[-1]
    b_down[-1] = b_up[-1]
    s_m_down.append(get_size(b_down[-1]))
    s_b_down.append(get_size(m_down[-1]))
    m_b_down.append(get_size(m_down[-1], bytes=True))
    for i in range(causal_graph.num_slices - 2, -1, -1):
        b_down[i] = m.insert(
            m.select(m_down[i + 1], causal_graph.get_slice_front_frontier(i)), b_up[i])
        s_b_down.append(get_size(b_down[i]))
        m_b_down.append(get_size(b_down[i], bytes=True))
        if i > 0:
            m_down[i] = m.insert(
                m.select(b_down[i], causal_graph.get_slice_back_frontier(i)), m_up[i])
            s_m_down.append(get_size(m_down[i]))
    
    return  s_b_i_border_tree, s_b_ip1_border_tree, s_m_up, s_b_up, s_m_down, s_b_down, m_b_up, m_b_down