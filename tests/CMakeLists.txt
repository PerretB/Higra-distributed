include_directories(${HIGRA_INCLUDE_DIRS} ${HIGRA_LIB_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS})

set(HIGRA_DISTRIBUTED_TESTS_PYTHON_SRC
        main.cpp
        ../higra_distributed/pydistributed.cpp)

add_executable(test_python_exe ${HIGRA_DISTRIBUTED_TESTS_PYTHON_SRC})

target_link_libraries(test_python_exe pybind11::embed)



set(UNIT_TEST_TARGETS ${UNIT_TEST_TARGETS} test_python_exe PARENT_SCOPE)

set(TEST_FILES
        test_extension.py)

foreach (FILE ${TEST_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()


add_test(NAME Test_python_cpp COMMAND  test_python_exe)
add_test(NAME Test_python
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys;\
                sys.path.insert(0, '${CMAKE_BINARY_DIR}');\
                import unittest;\
                result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}'));\
                exit(0 if result.wasSuccessful() else 1)"
        )

add_custom_target(all_tests ALL
        DEPENDS higra_distributedm test_python_exe
        )

        
if (DO_AUTO_TEST)
    add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND PYTHONPATH=${CMAKE_BINARY_DIR} ctest -V
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
endif ()
