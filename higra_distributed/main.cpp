#include "pybind11/pybind11.h"
#include "higra/graph.hpp"
#include "higra/sorting.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/accumulator/tree_accumulator.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"

namespace py = pybind11;


namespace hg {
    namespace distributed {
        template<typename T>
        auto select(const hg::tree &tree, const xt::xexpression<T> &xselected_leaves) {
            auto &selected_leaves = xselected_leaves.derived_cast();
            hg_assert_1d_array(selected_leaves);
            hg_assert_integral_value_type(selected_leaves);

            // indices of the nodes in the new tree, -1 if the node is not part of the new tree
            array_1d<index_t> new_node_index = xt::empty<index_t>({num_vertices(tree)});
            new_node_index.fill(-1);

            // at the beginning, all leaf nodes have:
            //  - a new node index of -1 if they are not selected
            //  - a temporary node index of 0 if they are selected
            // at the end of iteration i:
            //  - all nodes <= i have a valid new node index
            //  - all nodes > i that have a child <= i with a new node index != 1 have a temporary node index of 0
            xt::index_view(new_node_index, selected_leaves) = 0;
            index_t num_nodes_new_tree = 0;
            for (index_t i: leaves_to_root_iterator(tree, leaves_it::include, root_it::exclude)) {
                if (new_node_index(i) != -1) {
                    new_node_index(i) = num_nodes_new_tree;
                    new_node_index(parent(i, tree)) = 0;
                    ++num_nodes_new_tree;
                }
            }
            // special case for the root
            if (new_node_index(root(tree)) != -1) {
                new_node_index(root(tree)) = num_nodes_new_tree;
                ++num_nodes_new_tree;
            }

            // construct final parent and node map
            array_1d<index_t> new_parent = xt::empty<index_t>({num_nodes_new_tree});
            array_1d<index_t> node_map = xt::empty<index_t>({num_nodes_new_tree});

            index_t count = 0;
            for (index_t i: leaves_to_root_iterator(tree, leaves_it::include, root_it::include)) {
                if (new_node_index(i) != -1) {
                    new_parent(count) = new_node_index(parent(i, tree));
                    node_map(count) = i;
                    ++count;
                }
            }

            return std::make_pair(hg::tree(std::move(new_parent)), std::move(node_map));
        }

        template<typename T, typename T2>
        auto
        join(const hg::tree &tree1, const xt::xexpression<T> &xnode_map1, const xt::xexpression<T2> &xmst_weights1,
             const hg::tree &tree2, const xt::xexpression<T> &xnode_map2, const xt::xexpression<T2> &xmst_weights2,
             const xt::xexpression<T> &xborder_edge_sources, const xt::xexpression<T> &xborder_edge_targets,
             const xt::xexpression<T> &xborder_edge_map, const xt::xexpression<T2> &xborder_edge_weights) {
            auto &node_map1 = xnode_map1.derived_cast();
            auto &mst_weights1 = xmst_weights1.derived_cast();
            auto &node_map2 = xnode_map2.derived_cast();
            auto &mst_weights2 = xmst_weights2.derived_cast();
            auto &border_edge_sources = xborder_edge_sources.derived_cast();
            auto &border_edge_targets = xborder_edge_targets.derived_cast();
            auto &border_edge_map = xborder_edge_map.derived_cast();
            auto &border_edge_weights = xborder_edge_weights.derived_cast();
            hg_assert_1d_array(node_map1);
            hg_assert_1d_array(node_map2);
            hg_assert_1d_array(mst_weights1);
            hg_assert_1d_array(mst_weights2);
            hg_assert_1d_array(border_edge_sources);
            hg_assert_1d_array(border_edge_targets);
            hg_assert_1d_array(border_edge_map);
            hg_assert_1d_array(border_edge_weights);
            hg_assert_integral_value_type(node_map1);
            hg_assert_node_weights(tree1, node_map1);
            hg_assert_node_weights(tree2, node_map2);
            hg_assert(mst_weights1.size() == num_vertices(tree1) - num_leaves(tree1),
                      "mst altitudes 1 does not match with the size of tree1.");
            hg_assert(mst_weights2.size() == num_vertices(tree2) - num_leaves(tree2),
                      "mst altitudes 2 does not match with the size of tree2.");
            hg_assert_same_shape(border_edge_sources, border_edge_targets);
            hg_assert_same_shape(border_edge_sources, border_edge_targets);
            hg_assert_same_shape(border_edge_sources, border_edge_weights);

            const index_t size_tree1 = num_vertices(tree1);
            const index_t size_tree2 = num_vertices(tree2);
            const index_t num_leaves_tree1 = num_leaves(tree1);
            const index_t num_leaves_tree2 = num_leaves(tree2);
            index_t num_border_edges = border_edge_sources.size();
            index_t num_leaves_join = num_leaves_tree1 + num_leaves_tree2;

            using value_type = typename T2::value_type;
            using pair = std::pair<value_type, index_t>;
            pair infinity = pair((std::numeric_limits<value_type>::has_infinity) ?
                                 std::numeric_limits<value_type>::infinity() :
                                 std::numeric_limits<value_type>::max(),
                                 std::numeric_limits<index_t>::max());

            const auto less = [](const pair &p1, const pair &p2) {
                return (p1.first < p2.first) ||
                       ((p1.first == p2.first) && (p1.second < p2.second));
            };

            const auto attribute_child_one_leaf_node = [](const auto &tree, const index_t shift) {
                array_2d<index_t> attr({num_vertices(tree), 2}, -1);
                xt::noalias(xt::view(attr, xt::range(0, (index_t) num_leaves(tree)), 0)) =
                        xt::arange(shift, shift + (index_t) num_leaves(tree));

                for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude, root_it::include)) {
                    attr(i, 0) = attr(child(0, i, tree), 0);
                    if (num_children(i, tree) == 2) {
                        attr(i, 1) = attr(child(1, i, tree), 0);
                    }
                }
                return attr;
            };

            const auto child_one_leaf_tree1 = attribute_child_one_leaf_node(tree1, 0);
            const auto child_one_leaf_tree2 = attribute_child_one_leaf_node(tree2, num_leaves_tree1);

            const auto sorted_edge_indices = hg::stable_arg_sort(border_edge_weights);

            // disjoint forest
            union_find uf(num_leaves_join);

            // mapping canonical nodes of the disjoint forest to their respective tree root
            array_1d<index_t> roots = xt::arange<index_t>(num_leaves_join);

            // new parent mapping : size is the worst case
            array_1d<index_t> new_parents = xt::empty<index_t>({size_tree1 + size_tree2 + num_border_edges});
            // new node mapping, leaf nodes are mapped to their unique global graph vertex identifier
            // internal nodes are mapped to the unique global edge identifier of their mst building edge
            array_1d<index_t> new_node_map = xt::empty_like(new_parents);
            // initialize node mapping of leaves
            xt::noalias(xt::view(new_node_map, xt::range(0, num_leaves_tree1))) =
                    xt::view(node_map1, xt::range(0, num_leaves_tree1));
            xt::noalias(xt::view(new_node_map, xt::range(num_leaves_tree1, num_leaves_join))) =
                    xt::view(node_map2, xt::range(0, num_leaves_tree2));

            array_1d<value_type> new_mst_weights = xt::empty<value_type>({size_tree1 - num_leaves_tree1 +
                                                                          size_tree2 - num_leaves_tree2 +
                                                                          num_border_edges});

            // current number of nodes in the new tree
            index_t num_nodes = num_leaves_join;

            // current indices in the three edge sets
            index_t index_tree1 = num_leaves_tree1;
            index_t index_tree2 = num_leaves_tree2;
            index_t index_edge = 0;

            // while an edge set is not empty
            while (index_tree1 < size_tree1 || index_tree2 < size_tree2 || index_edge < num_border_edges) {
                const auto ei = sorted_edge_indices(index_edge);
                index_t canonical1 = -1;
                index_t canonical2 = -1;
                pair edge;


                pair iw_tree1 = (index_tree1 < size_tree1) ?
                                pair(mst_weights1(index_tree1 - num_leaves_tree1), node_map1(index_tree1)) :
                                infinity;
                pair iw_tree2 = (index_tree2 < size_tree2) ?
                                pair(mst_weights2(index_tree2 - num_leaves_tree2), node_map2(index_tree2)) :
                                infinity;
                pair iw_edge = (index_edge < num_border_edges) ?
                               pair(border_edge_weights(ei), border_edge_map(ei)) :
                               infinity;

                // smallest element is a border edge : normal Kruskal step
                if (less(iw_edge, iw_tree1) && less(iw_edge, iw_tree2)) {

                    auto source = border_edge_sources(ei);
                    auto target = border_edge_targets(ei);
                    canonical1 = uf.find(source);
                    canonical2 = uf.find(target + num_leaves_tree1);
                    edge = iw_edge;
                    ++index_edge;
                } else { // smallest element is a node of a tree

                    if (less(iw_tree1, iw_tree2)) {
                        edge = iw_tree1;
                        canonical1 = uf.find(child_one_leaf_tree1(index_tree1, 0));
                        if (child_one_leaf_tree1(index_tree1, 1) != -1) {
                            canonical2 = uf.find(child_one_leaf_tree1(index_tree1, 1));
                        }
                        ++index_tree1;
                    } else {
                        edge = iw_tree2;
                        canonical1 = uf.find(child_one_leaf_tree2(index_tree2, 0));
                        if (child_one_leaf_tree2(index_tree2, 1) != -1) {
                            canonical2 = uf.find(child_one_leaf_tree2(index_tree2, 1));
                        }
                        ++index_tree2;
                    }
                }

                if (canonical1 != canonical2) {
                    auto root1 = roots(canonical1);
                    auto new_node = num_nodes;
                    ++num_nodes;
                    new_parents[root1] = new_node;
                    new_mst_weights(new_node - (num_leaves_tree1 + num_leaves_tree2)) = edge.first;
                    new_node_map(new_node) = edge.second;

                    if (canonical2 == -1) {
                        roots(canonical1) = new_node;
                    } else {
                        auto root2 = roots(canonical2);
                        new_parents(root2) = new_node;
                        auto new_root = uf.link(canonical1, canonical2);
                        roots(new_root) = new_node;
                    }
                }
            }

            new_parents(num_nodes - 1) = num_nodes - 1;

            return std::make_tuple(
                    hg::tree(xt::view(new_parents, xt::range(0, num_nodes))),
                    xt::eval(xt::view(new_node_map, xt::range(0, num_nodes))),
                    xt::eval(xt::view(new_mst_weights,
                                      xt::range(0, num_nodes - (num_leaves_tree1 + num_leaves_tree2)))));
        }

        template<typename T, typename T2>
        auto
        insert(const hg::tree &tree1, const xt::xexpression<T> &xnode_map1, const xt::xexpression<T2> &xmst_weights1,
               const hg::tree &tree2, const xt::xexpression<T> &xnode_map2, const xt::xexpression<T2> &xmst_weights2) {
            auto &node_map1 = xnode_map1.derived_cast();
            auto &mst_weights1 = xmst_weights1.derived_cast();
            auto &node_map2 = xnode_map2.derived_cast();
            auto &mst_weights2 = xmst_weights2.derived_cast();
            hg_assert_1d_array(node_map1);
            hg_assert_1d_array(node_map2);
            hg_assert_1d_array(mst_weights1);
            hg_assert_1d_array(mst_weights2);
            hg_assert_integral_value_type(node_map1);
            hg_assert_node_weights(tree1, node_map1);
            hg_assert_node_weights(tree2, node_map2);
            hg_assert(mst_weights1.size() == num_vertices(tree1) - num_leaves(tree1),
                      "mst altitudes 1 size does not match with the size of tree1.");
            hg_assert(mst_weights2.size() == num_vertices(tree2) - num_leaves(tree2),
                      "mst altitudes 2 size does not match with the size of tree2.");

            using value_type = typename T2::value_type;
            using pair = std::pair<value_type, index_t>;
            pair infinity = pair((std::numeric_limits<value_type>::has_infinity) ?
                                 std::numeric_limits<value_type>::infinity() :
                                 std::numeric_limits<value_type>::max(),
                                 std::numeric_limits<index_t>::max());

            const auto less = [](const pair &p1, const pair &p2) {
                return (p1.first < p2.first) ||
                       ((p1.first == p2.first) && (p1.second < p2.second));
            };

            const index_t size_tree1 = num_vertices(tree1);
            const index_t size_tree2 = num_vertices(tree2);
            const index_t num_leaves_tree1 = num_leaves(tree1);
            const index_t num_leaves_tree2 = num_leaves(tree2);

            // will hold the index of each node in the merged tree
            array_1d<index_t> new_rank_tree1 = xt::zeros<index_t>({size_tree1});
            array_1d<index_t> new_rank_tree2 = xt::zeros<index_t>({size_tree2});

            // for each node of the merged tree, gives the corresding nodes from tree1 and tree2 (-1) if no such node exists
            array_2d<index_t> merge_tree_node_map = xt::empty<index_t>(
                    {(size_t) (size_tree1 + size_tree2), (size_t) 2});

            //
            //  Leaf processing
            //
            for (index_t i = 0, j = 0; i < num_leaves_tree2; ++i) {
                merge_tree_node_map(i, 1) = i;
                if(j < num_leaves_tree1 && node_map1(j) == node_map2(i)){
                    merge_tree_node_map(i, 0) = j;
                    ++j;
                }else{
                    merge_tree_node_map(i, 0) = -1;
                    // mark the parent of i as alive (cannot be a deleted root)
                    new_rank_tree2(parent(i, tree2)) = -1;
                }
            }

            //
            //  Main Loop
            //
            index_t index_tree1 = num_leaves_tree1;
            index_t index_tree2 = num_leaves_tree2;
            index_t current_rank = num_leaves_tree2;

            while (index_tree1 < size_tree1 || index_tree2 < size_tree2) {
                // followings nodes in tree1 and tree2 represent the same region: merge
                if (index_tree1 < size_tree1 && index_tree2 < size_tree2 &&
                    node_map1(index_tree1) == node_map2(index_tree2)) {
                    new_rank_tree1(index_tree1) = current_rank;
                    new_rank_tree2(index_tree2) = current_rank;
                    merge_tree_node_map(current_rank, 0) = index_tree1;
                    merge_tree_node_map(current_rank, 1) = index_tree2;
                    ++index_tree1;
                    ++index_tree2;
                    ++current_rank;
                } else {
                    pair iw_tree1 = (index_tree1 < size_tree1) ?
                                    pair(mst_weights1(index_tree1 - num_leaves_tree1), node_map1(index_tree1)) :
                                    infinity;
                    pair iw_tree2 = (index_tree2 < size_tree2) ?
                                    pair(mst_weights2(index_tree2 - num_leaves_tree2), node_map2(index_tree2)) :
                                    infinity;
                    if (less(iw_tree2, iw_tree1)) { // check condition
                        if (new_rank_tree2(index_tree2) != -1) {
                            new_rank_tree2(parent(index_tree2, tree2)) = -1;
                            new_rank_tree2(index_tree2) = current_rank;
                            merge_tree_node_map(current_rank, 0) = -1;
                            merge_tree_node_map(current_rank, 1) = index_tree2;
                            ++current_rank;
                        }
                        ++index_tree2;
                    } else {
                        new_rank_tree1(index_tree1) = current_rank;
                        merge_tree_node_map(current_rank, 0) = index_tree1;
                        merge_tree_node_map(current_rank, 1) = -1;
                        ++index_tree1;
                        ++current_rank;
                    }

                }
            }

            //
            // Merged tree creation
            //
            index_t num_nodes = current_rank;
            array_1d<index_t> new_parent = xt::empty<index_t>({num_nodes});
            array_1d<index_t> new_node_map = xt::empty<index_t>({num_nodes});
            array_1d<value_type> new_mst_weights = xt::empty<value_type>({num_nodes - num_leaves_tree2});

            for (index_t i = 0; i < num_nodes; ++i) {
                auto nt1 = merge_tree_node_map(i, 0);
                auto nt2 = merge_tree_node_map(i, 1);
                if (nt1 != -1) {
                    if (nt1 == root(tree1)) {
                        new_parent(i) = i;
                    } else {
                        new_parent(i) = new_rank_tree1(parent(nt1, tree1));
                    }
                    new_node_map(i) = node_map1(nt1);
                    if (i >= num_leaves_tree2) {
                        new_mst_weights(i - num_leaves_tree2) = mst_weights1(nt1 - num_leaves_tree1);
                    }
                } else {
                    if (nt2 == root(tree2)) {
                        new_parent(i) = i;
                    } else {
                        new_parent(i) = new_rank_tree2(parent(nt2, tree2));
                    }
                    new_node_map(i) = node_map2(nt2);
                    if (i >= num_leaves_tree2) {
                        new_mst_weights(i - num_leaves_tree2) = mst_weights2(nt2 - num_leaves_tree2);
                    }
                }
            }

            return std::make_tuple(hg::tree(std::move(new_parent)),
                                   std::move(new_node_map),
                                   std::move(new_mst_weights));
        }
    }
}


using namespace hg;


// Python Module and Docstrings
PYBIND11_MODULE(higra_distributedm, m) {
    xt::import_numpy();

    m.doc() = R"pbdoc(
        An example higra extension
    )pbdoc";

    m.def("_select", [](const hg::tree &tree, const xt::pyarray<hg::index_t> &selected_leaves) {
              hg_assert(xt::amin(selected_leaves)() >= 0 && xt::amax(selected_leaves)() < (index_t) num_leaves(tree),
                        "Invalid leaf index.");
              auto res = hg::distributed::select(tree, selected_leaves);
              return py::make_tuple(std::move(res.first), std::move(res.second));
          }, "Select a subtree induced by a subset of leaves of the input tree.",
          py::arg("tree"),
          py::arg("selected_leaves"));

    m.def("_join",
          [](const hg::tree &tree1, const xt::pyarray<hg::index_t> &node_map1, const xt::pyarray<double> &mst_weights1,
             const hg::tree &tree2, const xt::pyarray<hg::index_t> &node_map2, const xt::pyarray<double> &mst_weights2,
             const xt::pyarray<hg::index_t> &border_edge_sources, const xt::pyarray<hg::index_t> &border_edge_targets,
             const xt::pyarray<hg::index_t> &border_edge_map, const xt::pyarray<double> &border_edge_weights) {

              auto res = hg::distributed::join(tree1, node_map1, mst_weights1,
                                               tree2, node_map2, mst_weights2,
                                               border_edge_sources, border_edge_targets,
                                               border_edge_map, border_edge_weights);
              return py::make_tuple(std::move(std::get<0>(res)),
                                    std::move(std::get<1>(res)),
                                    std::move(std::get<2>(res)));
          }, "Join two trees by their border edges.");

    m.def("_insert",
          [](const hg::tree &tree1, const xt::pyarray<hg::index_t> &node_map1, const xt::pyarray<double> &mst_weights1,
             const hg::tree &tree2, const xt::pyarray<hg::index_t> &node_map2,
             const xt::pyarray<double> &mst_weights2) {

              auto res = hg::distributed::insert(tree1, node_map1, mst_weights1,
                                                 tree2, node_map2, mst_weights2);
              return py::make_tuple(std::move(std::get<0>(res)),
                                    std::move(std::get<1>(res)),
                                    std::move(std::get<2>(res)));
          }, "Insert tree1 into tree2.");

}
