/***************************************************************************
* Copyright ESIEE Paris (2021)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

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
        Higra extension for distributed binary partition tree
    )pbdoc";
    xt::import_numpy();
    py_init_distributed(m);
}
