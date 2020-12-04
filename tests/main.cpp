

/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#define CATCH_CONFIG_MAIN

//#include "xtl/xmeta_utils.hpp"
#define FORCE_IMPORT_ARRAY

#include "catch2/catch.hpp"
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <pybind11/embed.h> // everything needed for embedding
#include <iostream>
#include "../higra_distributed/pydistributed.hpp"
#include "higra/structure/array.hpp"
#include "higra/graph.hpp"

namespace py = pybind11;
using namespace py::literals;
//using namespace xt;
using namespace std;
using namespace hg;

// Python Module and Docstrings
PYBIND11_EMBEDDED_MODULE(higra_distributedmo, m) {
    m.doc() = R"pbdoc(
        An example higra extension
    )pbdoc";
    xt::import_numpy();
    py_init_distributed(m);
}

TEST_CASE("basic python embedded interpreter", "[embedded]") {
    py::scoped_interpreter guard{};
    SECTION("example import") {

        //py::initialize_interpreter();
        py::module higra = py::module::import("higra");
        py::module sys = py::module::import("higra_distributedmo");
        array_1d<index_t> par{3, 3, 4, 4, 4};
        tree t(par);
        array_1d<index_t> sel{0, 1};
        sys.attr("_select")(t, sel);
        //py::finalize_interpreter();
    }

    /*SECTION("example import2") {
        py::module sys = py::module::import("sys");
        auto tuple = sys.attr("path").cast<py::tuple>();
        cout << "PYTHON PATH !!!!!!!!!!!!!!!!!!!!!" << "\n";
        for(int i =0; i< tuple.size(); i++){
            cout << tuple[i].cast<string>() << "\n";
        }
        cout << "PYTHON PATH-------------------" << "\n";
    }*/


}

