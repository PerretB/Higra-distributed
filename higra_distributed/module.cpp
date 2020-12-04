#define FORCE_IMPORT_ARRAY
#include "pybind11/pybind11.h"

#include "xtl/xmeta_utils.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "pydistributed.hpp"

namespace py = pybind11;

// Python Module and Docstrings
PYBIND11_MODULE(higra_distributedm, m) {
    m.doc() = R"pbdoc(
        An example higra extension
    )pbdoc";
    xt::import_numpy();
    py_init_distributed(m);
}
