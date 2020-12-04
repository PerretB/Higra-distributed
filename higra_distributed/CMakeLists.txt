include_directories(${HIGRA_INCLUDE_DIRS} ${HIGRA_LIB_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

function(REGISTER_PYTHON_MODULE_FILES PYTHON_FILES)
    foreach (FILE ${PYTHON_FILES})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
    endforeach ()
endfunction(REGISTER_PYTHON_MODULE_FILES)

set(PY_FILES
        __init__.py
        distributed.py)

REGISTER_PYTHON_MODULE_FILES("${PY_FILES}")

set(PYMODULE_COMPONENTS
        pydistributed.cpp
        module.cpp)

pybind11_add_module(higra_distributedm ${PYMODULE_COMPONENTS})


