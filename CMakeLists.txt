

cmake_minimum_required(VERSION 2.8.12)
project(higra-distributed)



##########################
#  CMAKE config
##########################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(Higra REQUIRED)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${HIGRA_LIB_CMAKE_DIRS})


##########################
#  Build Type and flags
##########################

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."  FORCE)
endif()
message("Current build type is: ${CMAKE_BUILD_TYPE}")

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 14)

message("Current compiler is: ${CMAKE_CXX_COMPILER_ID}")

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
    endif()
	
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs")

    #if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND UNIX AND NOT APPLE)
    #    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fclang-abi-compat=8 -Wno-missing-braces")
    #endif()
	
	if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=8")
    endif()

    

    # Not debug => optim flag
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ffast-math -Wno-strict-overflow")#-march=native

        # check link time optimization -faligned-new
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        endif()
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif ()
        endif ()
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # using Visual Studio C++
    CHECK_CXX_COMPILER_FLAG("/std:c++14" COMPILER_SUPPORTS_CXX14)
    if(COMPILER_SUPPORTS_CXX14)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
    endif()

    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /GL /EHsc /MP /bigobj /O2")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG /MANIFEST:NO")
    endif()
endif()

message("CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})
##########################
#  Options
##########################


enable_testing()

option (USE_SIMD
        "Enabled SIMD support for xtensor." ON)

if (USE_SIMD)
    add_definitions("-DXTENSOR_USE_XSIMD")
endif()

if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES RELEASE)
    add_definitions("-DHG_ENABLE_TRACE")
endif()

##########################
#  Build Python bindings
##########################

find_package(PythonInterp 3 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(NumPy REQUIRED)

add_subdirectory(higra_distributed)

configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(README.md ${CMAKE_CURRENT_BINARY_DIR}/README.md COPYONLY)

##########################
#  Build Tests
##########################
option(DO_AUTO_TEST
        "Execute test suit automatically at the end of the build." ON)
add_subdirectory(tests)


