# - Find the Higra libraries
# This module finds if Higra is installed, and sets the following variables
# indicating where it is.
#
#  HIGRA_FOUND               - was Higra found
#  HIGRA_VERSION             - the version of Higra found as a string
#  HIGRA_VERSION_MAJOR       - the major version number of Higra
#  HIGRA_VERSION_MINOR       - the minor version number of Higra
#  HIGRA_VERSION_PATCH       - the patch version number of Higra
#  HIGRA_INCLUDE_DIRS        - path to the Higra include files
#  HIGRA_LIB_INCLUDE_DIRS    - path to the Higra third party libraries include files
#  HIGRA_LIB_CMAKE_DIRS      - path to the Higra third party cmake files

#============================================================================


# Finding Higra involves calling the Python interpreter
if(Higra_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
else()
    find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
    set(HIGRA_FOUND FALSE)
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import higra as hg; print(hg.version()); print(hg.get_include()); print(hg.get_lib_include()); print(hg.get_lib_cmake());"
    RESULT_VARIABLE _HIGRA_SEARCH_SUCCESS
    OUTPUT_VARIABLE _HIGRA_VALUES
    ERROR_VARIABLE _HIGRA_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _HIGRA_SEARCH_SUCCESS MATCHES 0)
    if(NumPy_FIND_REQUIRED)
        message(FATAL_ERROR
            "Higra import failure:\n${_HIGRA_ERROR_VALUE}")
    endif()
    set(HIGRA_FOUND FALSE)
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _HIGRA_VALUES ${_HIGRA_VALUES})
string(REGEX REPLACE "\n" ";" _HIGRA_VALUES ${_HIGRA_VALUES})
list(GET _HIGRA_VALUES 0 HIGRA_VERSION)
list(GET _HIGRA_VALUES 1 HIGRA_INCLUDE_DIRS)
list(GET _HIGRA_VALUES 2 HIGRA_LIB_INCLUDE_DIRS)
list(GET _HIGRA_VALUES 3 HIGRA_LIB_CMAKE_DIRS)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" HIGRA_INCLUDE_DIRS ${HIGRA_INCLUDE_DIRS})
string(REGEX REPLACE "\\\\" "/" HIGRA_LIB_INCLUDE_DIRS ${HIGRA_LIB_INCLUDE_DIRS})
string(REGEX REPLACE "\\\\" "/" HIGRA_LIB_CMAKE_DIRS ${HIGRA_LIB_CMAKE_DIRS})

# Get the major and minor version numbers
string(REGEX REPLACE "\\." ";" _HIGRA_VERSION_LIST ${HIGRA_VERSION})
list(GET _HIGRA_VERSION_LIST 0 HIGRA_VERSION_MAJOR)
list(GET _HIGRA_VERSION_LIST 1 HIGRA_VERSION_MINOR)
list(GET _HIGRA_VERSION_LIST 2 HIGRA_VERSION_PATCH)
string(REGEX MATCH "[0-9]*" HIGRA_VERSION_PATCH ${HIGRA_VERSION_PATCH})


find_package_message(HIGRA
    "Found Higra: version \"${HIGRA_VERSION}\" ${HIGRA_INCLUDE_DIRS}"
    "${HIGRA_INCLUDE_DIRS}${HIGRA_VERSION}")

set(HIGRA_FOUND TRUE)
