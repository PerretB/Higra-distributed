#include "pybind11/pybind11.h"
#include "higra/graph.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

// Example
auto example_function(const hg::tree & tree, const xt::pyarray<double> & altitudes){
    xt::pyarray<double> res = xt::zeros_like(altitudes);
    for(auto n: leaves_to_root_iterator(tree, hg::leaves_it::exclude)){
        double tmp = 1;
        for(auto c: hg::children_iterator(n, tree)){
            tmp *= res[c];
        }
        res[n] = altitudes[n] + tmp;
    }
    return res;
}

// Python Module and Docstrings
PYBIND11_MODULE(higra_distributed, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        An example higra extension
    )pbdoc";

    m.def("example_function", example_function, "An example function.");
    
}
