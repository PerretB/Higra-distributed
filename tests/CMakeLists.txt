set(TEST_FILES
        test_extension.py)

foreach (FILE ${TEST_FILES})
    configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endforeach ()

add_test(NAME Test_python
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys;\
                sys.path.insert(0, '${CMAKE_BINARY_DIR}');\
                import unittest;\
                result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('${CMAKE_CURRENT_BINARY_DIR}'));\
                exit(0 if result.wasSuccessful() else 1)"
        )

add_custom_target(all_tests ALL
        DEPENDS higra_distributedm
        )

        
if (DO_AUTO_TEST)
    add_custom_command(TARGET all_tests
            COMMENT "Run tests"
            POST_BUILD COMMAND PYTHONPATH=${CMAKE_BINARY_DIR} ctest -V
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            )
endif ()
