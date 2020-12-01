include_directories(${HIGRA_INCLUDE_DIRS} ${HIGRA_LIB_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

set(PYMODULE_COMPONENTS
        main.cpp)

pybind11_add_module(higra_distributed ${PYMODULE_COMPONENTS})


