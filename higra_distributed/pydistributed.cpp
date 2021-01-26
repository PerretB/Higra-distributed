#include "pydistributed.hpp"

#include "pybind11/pybind11.h"

#include "higra/distributed/core.hpp"

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;

using namespace hg;

void py_init_distributed(pybind11::module &m) {
    //xt::import_numpy();

    m.def("_select", [](const hg::tree &tree,
                        const xt::pyarray<hg::index_t> &node_map,
                        const xt::pyarray<double> &mst_weights,
                        const xt::pyarray<hg::index_t> &selected_vertices) {
              auto res = hg::distributed::select(tree, node_map, mst_weights, selected_vertices);
              return py::make_tuple(std::move(std::get<0>(res)),
                                    std::move(std::get<1>(res)),
                                    std::move(std::get<2>(res)));
          }, "Select a subtree induced by a subset of leaves of the input tree.",
          py::arg("tree"),
          py::arg("node_map"),
          py::arg("mst_weights"),
          py::arg("selected_vertices"));

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

